"""
재학습 파이프라인 (챔피언 / 챌린저)

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
  5) 교체 - 챌린저가 더 좋으면 전체 데이터로 다시 학습해 배포
     (models/ 에 버전별 보관, price_model.joblib / recommend_pool.csv 갱신)
  모든 실행 결과(건너뜀 포함)는 models/registry.jsonl 에 한 줄씩 남는다.

실행 (크롤링 01 → 정답 수집 03 다음에)
    python 04_retrain.py           # 트리거가 없으면 건너뜀
    python 04_retrain.py --force   # 트리거와 무관하게 재학습/비교
교체된 모델은 app.py 를 재시작하면 반영된다.
"""

import argparse
import hashlib
import importlib.util
import json
import shutil
from datetime import datetime, timezone
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


def model_version(path: Path) -> str:
    """app.py 의 MODEL_VERSION 과 같은 방식 (로그의 model_version 과 맞춰 보기 위함)."""
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


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
    X_cols = state["columns"]
    joblib.dump({
        "model": model,
        "feature_state": state,
        "feature_columns": X_cols,
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="트리거와 무관하게 재학습/비교")
    args = parser.parse_args()

    champion = joblib.load(PROD_MODEL)
    champ_ver = model_version(PROD_MODEL)
    trained_max_no = champion.get("trained_max_no")
    if trained_max_no is None:
        raise SystemExit("현재 모델에 trained_max_no 가 없습니다. 02_price_regression.py 를 한 번 다시 실행해 주세요.")

    df = m.load_and_clean(m.CSV_PATH)
    y = df[m.TARGET_COL]
    nos = m.listing_no(df["링크"])
    is_new = (nos > trained_max_no).to_numpy()
    n_new = int(is_new.sum())

    # 1) 트리거 --------------------------------------------------------------
    lift, n_labels = online_lift(champ_ver)
    triggers = []
    if n_new >= MIN_NEW_ROWS:
        triggers.append(f"데이터: 새 매물 {n_new}건")
    if lift is not None and lift <= 0:
        triggers.append(f"성능: 저렴 추천 판매율 우위 {lift:+.3f}")
    if args.force:
        triggers.append("수동(--force)")

    print(f"챔피언 {champ_ver} (학습 {champion.get('n_train')}건, 마지막 매물 no={trained_max_no})")
    print(f"새 매물 {n_new}건 / 로그 확정 정답 {n_labels}건"
          + (f", 저렴 추천 판매율 우위 {lift:+.3f}" if lift is not None else " (성능 판단에는 부족)"))
    base = {"champion": champ_ver, "n_new": n_new, "n_labels": n_labels, "online_lift": lift}
    if not triggers:
        print("트리거 없음 -> 재학습 건너뜀")
        record({**base, "decision": "skip", "reason": "트리거 없음"})
        return
    print("트리거: " + ", ".join(triggers))

    # 2) 시간순 분리: 새 매물 중 가장 최근 것을 시험용으로 ----------------------
    new_idx = np.flatnonzero(is_new)
    new_idx = new_idx[np.argsort(nos.to_numpy()[new_idx])]
    n_hold = int(len(new_idx) * HOLDOUT_FRAC)
    if n_hold < MIN_HOLDOUT:
        print(f"시험용 새 매물이 {n_hold}건뿐이라 공정한 비교가 불가능 -> 교체하지 않음")
        record({**base, "triggers": triggers, "decision": "keep", "reason": f"시험용 부족({n_hold}건)"})
        return
    hold = np.zeros(len(df), dtype=bool)
    hold[new_idx[-n_hold:]] = True

    # 3) 챌린저 학습 (시험용 제외) ---------------------------------------------
    train_df = df[~hold].reset_index(drop=True)
    test_df = df[hold].reset_index(drop=True)
    state = m.fit_feature_state(train_df)
    challenger = m.make_gbr().fit(m.transform_features(train_df, state), train_df[m.TARGET_COL])

    # 4) 같은 시험용으로 비교 --------------------------------------------------
    y_test = test_df[m.TARGET_COL]
    pred_champ = champion["model"].predict(m.transform_features(test_df, champion["feature_state"]))
    pred_chall = challenger.predict(m.transform_features(test_df, state))
    metrics = {
        "champion": {"mae": mean_absolute_error(y_test, pred_champ), "r2": r2_score(y_test, pred_champ)},
        "challenger": {"mae": mean_absolute_error(y_test, pred_chall), "r2": r2_score(y_test, pred_chall)},
    }
    print(f"\n시험용(가장 최근 새 매물) {n_hold}건 비교")
    for k, v in metrics.items():
        print(f"  {k:10s} MAE {v['mae']:.3f}  R² {v['r2']:.3f}")

    if metrics["challenger"]["mae"] >= metrics["champion"]["mae"]:
        print("챌린저가 더 낫지 않음 -> 챔피언 유지")
        record({**base, "triggers": triggers, "n_holdout": n_hold, "metrics": metrics, "decision": "keep",
                "reason": "챌린저 MAE가 더 낮지 않음"})
        return

    # 5) 교체: 전체 데이터로 다시 학습해 배포 ----------------------------------
    MODELS_DIR.mkdir(exist_ok=True)
    archived = MODELS_DIR / f"model_{champ_ver}.joblib"
    if not archived.exists():
        shutil.copy2(PROD_MODEL, archived)  # 롤백용으로 이전 챔피언 보관

    full_state = m.fit_feature_state(df)
    X_all = m.transform_features(df, full_state)
    final = m.make_gbr().fit(X_all, y)
    tmp = MODELS_DIR / "model_new.joblib"
    save_model(final, full_state, df, tmp)
    new_ver = model_version(tmp)
    versioned = MODELS_DIR / f"model_{new_ver}.joblib"
    tmp.replace(versioned)
    shutil.copy2(versioned, PROD_MODEL)

    pool = m.build_recommend_pool(df, X_all, y, m.make_gbr)
    pool.to_csv(m.POOL_CSV, index=False, encoding="utf-8-sig")

    print(f"\n챌린저로 교체: {champ_ver} -> {new_ver} (이전 모델 보관: {archived})")
    print(f"{PROD_MODEL}, {m.POOL_CSV} 갱신 완료. app.py 를 재시작하면 반영됩니다.")
    record({**base, "triggers": triggers, "n_holdout": n_hold, "metrics": metrics, "decision": "promote",
            "new_version": new_ver, "n_train": len(df)})


if __name__ == "__main__":
    main()
