"""
재학습 파이프라인 (데이터 검증 + 챔피언 / 챌린저 + 추천 후보 갱신)

  0) 데이터 검증 - 마지막 크롤링 결과가 정상인지 먼저 확인. 이상하면 여기서 멈추고
     (재학습도, 추천 후보 갱신도 하지 않음) 종료 코드 1로 끝나 스케줄러가 실패를 알 수 있게 함.
     사이트 구조가 바뀌어 크롤링이 반쯤 깨진 데이터로 모델/추천이 바뀌는 것을 막기 위함.
  1) 트리거 확인 - 아래 중 하나라도 해당하면 재학습
     - 데이터 트리거: 현재 모델(챔피언)이 학습한 뒤 새로 올라온 매물이 MIN_NEW_ROWS건 이상
       (매물 번호 no 가 챔피언의 trained_max_no 보다 큰 매물 = 챔피언이 본 적 없는 매물)
     - 성능 트리거: 요청 로그 + 정답(03_collect_labels.py 결과)에서, 챔피언이 '저렴'이라고
       추천한 매물의 판매율이 '시세 수준' 매물보다 높지 않음 (확정 정답 MIN_LABELS건 이상일 때)
     - --force
  2) 시간순 분리 - 새 매물 중 가장 최근 HOLDOUT_FRAC 를 시험용으로 떼어 둠
     (무작위로 나누면 챔피언이 이미 학습한 매물이 시험용에 섞여 챔피언이 부당하게 유리해짐)
  3) 챌린저 학습 - 시험용을 뺀 나머지 전체(기존 매물 + 앞쪽 새 매물)
  4) 비교 - 같은 시험용 매물로 챔피언 vs 챌린저 MAE
  5) 교체 - 챌린저가 더 좋으면 전체 데이터로 다시 학습해 배포 (models/ 에 버전별 보관)
  6) 추천 후보 갱신 - 교체 여부와 관계없이 최신 크롤링으로 recommend_pool.csv 를 다시 만듦
     (안 그러면 모델이 유지되는 날에는 이미 팔린 매물을 계속 추천하게 됨)
  모든 실행 결과(검증 실패/건너뜀/유지/교체)는 models/registry.jsonl 에 한 줄씩 남는다.
  실행 중인 app.py 는 모델/후보 파일이 바뀐 것을 감지해 재시작 없이 다시 읽는다.

실행 (크롤링 01 → 정답 수집 03 다음에)
    python 04_retrain.py           # 트리거가 없으면 재학습은 건너뛰고 추천 후보만 갱신
    python 04_retrain.py --force   # 트리거와 무관하게 재학습/비교
"""

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

_spec = importlib.util.spec_from_file_location(
    "price_regression", Path(__file__).resolve().parent / "02_price_regression.py")
m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m)

PROD_MODEL = Path("price_model.joblib")
MODELS_DIR = Path("models")
REGISTRY = MODELS_DIR / "registry.jsonl"
LABELS_CSV = Path("logs/recommend_labels.csv")

MIN_NEW_ROWS = 50     # 데이터 트리거 기준
MIN_LABELS = 50       # 성능 트리거를 판단할 최소 확정 정답 수
HOLDOUT_FRAC = 0.5    # 새 매물 중 시험용 비율 (가장 최근 순)
MIN_HOLDOUT = 20      # 시험용이 이보다 적으면 비교 결과를 믿을 수 없어 교체하지 않음

# 데이터 검증 기준. 2026-10 정상 크롤링 기준값(괄호)의 대략 절반 수준으로 잡음.
#   (최근 수집 3,222건 / 일반 매물 중 신차대비가격 있음 29.5%, 가격 있음 49.7%,
#    신차대비가격 중앙값 45%, 가격 중앙값 2,890만원, 브랜드 58개)
CHECKS = {
    "max_age_days": 2,            # 마지막 크롤링이 오늘 기준 며칠 이내여야 하는지
    "min_rows": 1500,             # 마지막 크롤링 매물 수
    "min_ratio_share": 0.15,      # 일반 매물 중 신차대비가격이 있는 비율
    "min_price_share": 0.25,      # 일반 매물 중 가격이 숫자인 비율
    "ratio_median": (20, 80),     # 신차대비가격(%) 중앙값 허용 범위
    "price_median": (5_000_000, 100_000_000),
    "max_missing_core": 0.05,     # 연식/주행거리/배기량 결측 비율
    "min_brands": 20,
}


# ---------------------------------------------------------------------------
# 0) 데이터 검증
# ---------------------------------------------------------------------------
def validate_data(csv_path: str) -> tuple[list[str], dict]:
    """마지막 크롤링분을 점검해 (문제 목록, 통계)를 돌려준다. 문제 목록이 비어 있으면 통과."""
    raw = pd.read_csv(csv_path)
    latest = str(raw["수집일"].max())
    cur = raw[raw["수집일"] == latest]
    normal = cur[cur["매물유형"] == "일반"] if "매물유형" in cur.columns else cur
    ratio = pd.to_numeric(normal["신차대비가격"], errors="coerce")
    price = pd.to_numeric(normal["가격"], errors="coerce")
    core_missing = max(pd.to_numeric(cur[c], errors="coerce").isna().mean() for c in ["연식", "주행거리", "배기량"])

    stats = {
        "latest_crawl": latest,
        "rows": len(cur),
        "ratio_share": round(float(ratio.notna().mean()), 3) if len(normal) else 0.0,
        "price_share": round(float(price.notna().mean()), 3) if len(normal) else 0.0,
        "ratio_median": float(ratio.median()) if ratio.notna().any() else None,
        "price_median": float(price.median()) if price.notna().any() else None,
        "missing_core": round(float(core_missing), 3),
        "brands": int(cur["브랜드"].nunique()),
    }

    c = CHECKS
    problems = []
    if date.fromisoformat(latest) < date.today() - timedelta(days=c["max_age_days"]):
        problems.append(f"마지막 크롤링이 오래됨({latest}) - 01 크롤링이 실패했을 수 있음")
    if stats["rows"] < c["min_rows"]:
        problems.append(f"매물 수 {stats['rows']}건 < {c['min_rows']}")
    if stats["ratio_share"] < c["min_ratio_share"]:
        problems.append(f"신차대비가격 있는 비율 {stats['ratio_share']} < {c['min_ratio_share']}")
    if stats["price_share"] < c["min_price_share"]:
        problems.append(f"가격 있는 비율 {stats['price_share']} < {c['min_price_share']}")
    lo, hi = c["ratio_median"]
    if stats["ratio_median"] is None or not lo <= stats["ratio_median"] <= hi:
        problems.append(f"신차대비가격 중앙값 {stats['ratio_median']} 범위({lo}~{hi}) 밖")
    lo, hi = c["price_median"]
    if stats["price_median"] is None or not lo <= stats["price_median"] <= hi:
        problems.append(f"가격 중앙값 {stats['price_median']} 범위({lo:,}~{hi:,}) 밖")
    if stats["missing_core"] > c["max_missing_core"]:
        problems.append(f"연식/주행거리/배기량 결측 {stats['missing_core']} > {c['max_missing_core']}")
    if stats["brands"] < c["min_brands"]:
        problems.append(f"브랜드 수 {stats['brands']} < {c['min_brands']}")
    return problems, stats


# ---------------------------------------------------------------------------
# 공통 도구
# ---------------------------------------------------------------------------
def model_version(path: Path) -> str:
    """app.py 의 MODEL_VERSION 과 같은 방식 (로그의 model_version 과 맞춰 보기 위함)."""
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def replace_atomic(src: Path, dst: Path):
    """dst 를 한 번에 교체. 실행 중인 app.py 가 쓰다 만 파일을 읽지 않도록 같은 폴더의
    임시 파일에 복사한 뒤 os.replace 로 바꿔치기한다."""
    tmp = dst.with_name(dst.name + ".tmp")
    shutil.copy2(src, tmp)
    os.replace(tmp, dst)


def online_lift(champion_version: str):
    """로그 기반 운영 성능: 챔피언의 '저렴' 추천 판매율 - '시세 수준' 추천 판매율.
    정답이 부족하면 None."""
    if not LABELS_CSV.exists():
        return None, 0
    lab = pd.read_csv(LABELS_CSV)
    lab = lab[(lab["model_version"] == champion_version) & (lab["판매여부"] != "미확정")]
    sold = lab["판매여부"].eq("판매됨")
    rates = sold.groupby(lab["저렴"].astype(bool)).mean()
    if len(lab) < MIN_LABELS or len(rates) < 2:
        return None, len(lab)
    return float(rates[True] - rates[False]), len(lab)


def save_model(model, state, df, path: Path):
    """02_price_regression.py 가 저장하는 형식과 같게 저장 (app.py 가 그대로 읽음)."""
    joblib.dump({
        "model": model,
        "feature_state": state,
        "feature_columns": state["columns"],
        "target_col": m.TARGET_COL,
        "ref_year_month": (m.REF_YEAR, m.REF_MONTH),
        "trained_max_no": int(m.listing_no(df["링크"]).max()),
        "n_train": len(df),
    }, path)


def record(entry: dict):
    MODELS_DIR.mkdir(exist_ok=True)
    entry = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), **entry}
    with REGISTRY.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
    print(f"\n레지스트리 기록: {REGISTRY}\n{json.dumps(entry, ensure_ascii=False, indent=2, default=str)}")


# ---------------------------------------------------------------------------
# 1)~5) 재학습 판단
# ---------------------------------------------------------------------------
def retrain(df: pd.DataFrame, force: bool) -> dict:
    """트리거 확인 -> 챌린저 학습/비교 -> 필요하면 교체. 레지스트리에 남길 내용을 돌려준다."""
    champion = joblib.load(PROD_MODEL)
    champ_ver = model_version(PROD_MODEL)
    trained_max_no = champion.get("trained_max_no")
    if trained_max_no is None:
        raise SystemExit("현재 모델에 trained_max_no 가 없습니다. 02_price_regression.py 를 한 번 다시 실행해 주세요.")

    y = df[m.TARGET_COL]
    nos = m.listing_no(df["링크"])
    is_new = (nos > trained_max_no).to_numpy()
    n_new = int(is_new.sum())

    lift, n_labels = online_lift(champ_ver)
    triggers = []
    if n_new >= MIN_NEW_ROWS:
        triggers.append(f"데이터: 새 매물 {n_new}건")
    if lift is not None and lift <= 0:
        triggers.append(f"성능: 저렴 추천 판매율 우위 {lift:+.3f}")
    if force:
        triggers.append("수동(--force)")

    print(f"챔피언 {champ_ver} (학습 {champion.get('n_train')}건, 마지막 매물 no={trained_max_no})")
    print(f"새 매물 {n_new}건 / 로그 확정 정답 {n_labels}건"
          + (f", 저렴 추천 판매율 우위 {lift:+.3f}" if lift is not None else " (성능 판단에는 부족)"))
    entry = {"champion": champ_ver, "n_new": n_new, "n_labels": n_labels, "online_lift": lift}
    if not triggers:
        print("트리거 없음 -> 재학습 건너뜀")
        return {**entry, "decision": "skip", "reason": "트리거 없음"}
    print("트리거: " + ", ".join(triggers))
    entry["triggers"] = triggers

    # 2) 시간순 분리: 새 매물 중 가장 최근 것을 시험용으로
    new_idx = np.flatnonzero(is_new)
    new_idx = new_idx[np.argsort(nos.to_numpy()[new_idx])]
    n_hold = int(len(new_idx) * HOLDOUT_FRAC)
    if n_hold < MIN_HOLDOUT:
        print(f"시험용 새 매물이 {n_hold}건뿐이라 공정한 비교가 불가능 -> 교체하지 않음")
        return {**entry, "decision": "keep", "reason": f"시험용 부족({n_hold}건)"}
    hold = np.zeros(len(df), dtype=bool)
    hold[new_idx[-n_hold:]] = True
    entry["n_holdout"] = n_hold

    # 3) 챌린저 학습 (시험용 제외)
    train_df = df[~hold].reset_index(drop=True)
    test_df = df[hold].reset_index(drop=True)
    state = m.fit_feature_state(train_df)
    challenger = m.make_gbr().fit(m.transform_features(train_df, state), train_df[m.TARGET_COL])

    # 4) 같은 시험용으로 비교
    y_test = test_df[m.TARGET_COL]
    pred_champ = champion["model"].predict(m.transform_features(test_df, champion["feature_state"]))
    pred_chall = challenger.predict(m.transform_features(test_df, state))
    metrics = {
        "champion": {"mae": mean_absolute_error(y_test, pred_champ), "r2": r2_score(y_test, pred_champ)},
        "challenger": {"mae": mean_absolute_error(y_test, pred_chall), "r2": r2_score(y_test, pred_chall)},
    }
    entry["metrics"] = metrics
    print(f"\n시험용(가장 최근 새 매물) {n_hold}건 비교")
    for k, v in metrics.items():
        print(f"  {k:10s} MAE {v['mae']:.3f}  R² {v['r2']:.3f}")

    if metrics["challenger"]["mae"] >= metrics["champion"]["mae"]:
        print("챌린저가 더 낫지 않음 -> 챔피언 유지")
        return {**entry, "decision": "keep", "reason": "챌린저 MAE가 더 낮지 않음"}

    # 5) 교체: 전체 데이터로 다시 학습해 배포
    MODELS_DIR.mkdir(exist_ok=True)
    archived = MODELS_DIR / f"model_{champ_ver}.joblib"
    if not archived.exists():
        shutil.copy2(PROD_MODEL, archived)  # 롤백용으로 이전 챔피언 보관

    full_state = m.fit_feature_state(df)
    final = m.make_gbr().fit(m.transform_features(df, full_state), y)
    tmp = MODELS_DIR / "model_new.joblib"
    save_model(final, full_state, df, tmp)
    new_ver = model_version(tmp)
    versioned = MODELS_DIR / f"model_{new_ver}.joblib"
    tmp.replace(versioned)
    replace_atomic(versioned, PROD_MODEL)

    print(f"\n챌린저로 교체: {champ_ver} -> {new_ver} (이전 모델 보관: {archived})")
    return {**entry, "decision": "promote", "new_version": new_ver, "n_train": len(df)}


# ---------------------------------------------------------------------------
# 6) 추천 후보 갱신
# ---------------------------------------------------------------------------
def refresh_pool(df: pd.DataFrame) -> dict:
    """최신 크롤링 데이터로 추천 후보와 적정가를 다시 만든다 (02 와 같은 out-of-fold 방식)."""
    state = m.fit_feature_state(df)
    pool = m.build_recommend_pool(df, m.transform_features(df, state), df[m.TARGET_COL], m.make_gbr)
    tmp = Path(m.POOL_CSV + ".new")
    pool.to_csv(tmp, index=False, encoding="utf-8-sig")
    os.replace(tmp, m.POOL_CSV)
    info = {"rows": len(pool), "cheap": int(pool["저렴"].sum()), "suspect": int(pool["이상매물의심"].sum())}
    print(f"\n추천 후보 갱신: {m.POOL_CSV} ({info['rows']}건, 저렴 {info['cheap']}건, 이상 매물 의심 {info['suspect']}건)")
    return info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="트리거와 무관하게 재학습/비교")
    args = parser.parse_args()

    problems, stats = validate_data(m.CSV_PATH)
    print(f"데이터 검증: {json.dumps(stats, ensure_ascii=False)}")
    if problems:
        print("데이터 검증 실패 -> 재학습과 추천 후보 갱신을 모두 중단합니다:")
        for p in problems:
            print(f"  - {p}")
        record({"decision": "blocked", "problems": problems, "data_stats": stats})
        sys.exit(1)
    print("데이터 검증 통과\n")

    df = m.load_and_clean(m.CSV_PATH)
    entry = retrain(df, args.force)
    entry["pool"] = refresh_pool(df)
    entry["data_stats"] = stats
    record(entry)


if __name__ == "__main__":
    main()
