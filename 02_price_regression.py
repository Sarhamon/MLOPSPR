"""
중고차 신차대비가격 예측 모델 (파이썬 버전, R 불필요) - 성능 개선판

기존 버전(랜덤포레스트, 원래 특성)에서 바뀐 점
  1) 연식 보정   : 01 스크립트가 만든 '연식' 숫자는 월(month)이 반대 방향으로
                   더해져 있어(12*(2022-년)+월) 같은 해 안에서 순서가 뒤집힙니다.
                   숫자에서 년/월을 복원해 '나이(개월)'을 새로 계산합니다.
  2) 특성 추가   : 브랜드, 이름(트림/세대 등) 단어, log 주행거리, 연당 주행거리
  3) 모델 추가   : GradientBoosting(huber 손실) - 이상치에 덜 흔들리는 부스팅
  4) 평가 개선   : 한 번 나눈 결과(운에 좌우됨) + 5-fold x 3회 반복 교차검증 병기
  5) 그래프 한글 : 한글 폰트 자동 지정 (글자 깨짐 해결)

실행 순서
    1) cars_processed.csv 가 이 파일과 같은 폴더에 있어야 합니다.
       (01_crawl_and_preprocess.py 는 데이터를 새로 받을 때만 실행)
    2) python -m pip install pandas numpy scikit-learn joblib matplotlib
    3) python 02_price_regression.py
"""

import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # 화면 없는 환경(서버 등)에서도 그래프 파일 저장 가능
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# 한글 폰트: 설치된 것 중 앞에서부터 처음 발견되는 폰트를 사용 (Windows/Mac/Linux 공통)
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = [
    "Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans CJK KR", "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False

# ---------------------------------------------------------------------------
# 설정
# ---------------------------------------------------------------------------
CSV_PATH = "cars_processed.csv"
POOL_CSV = "recommend_pool.csv"  # 추천 후보 매물 + 적정 가격 (recommender.py / app.py 가 사용)
SUSPECT_DISCOUNT = 0.5  # 적정가 대비 이 비율 넘게 싼 매물은 이상 매물로 보고 추천 제외

# '저렴' 판정용 비교 그룹 (build_recommend_pool 참고)
DOMESTIC_BRANDS = {"현대", "기아", "제네시스", "쉐보레", "KG모빌리티", "르노코리아", "르노삼성", "GM대우", "쌍용", "대우"}
BARGAIN_AGE_BINS = [-1, 3, 6, 10, 15, 100]  # 연식대(년): ~3, 3~6, 6~10, 10~15, 15~
BARGAIN_QUANTILE = 0.1  # 그룹 안에서 하위 10% 가격이면 '저렴'
MIN_GROUP_SIZE = 30     # 이보다 작은 그룹은 전체 기준 사용
TARGET_COL = "신차대비가격"   # 예측 대상 (%, 중고가 / 신차가 * 100)

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_SPLITS, CV_REPEATS = 5, 3

# 나이(개월) 계산 기준 시점. 나이는 이 값만큼 평행이동할 뿐이라 예측에는 영향이
# 거의 없고, 값을 고정해 두면 언제 실행해도 결과가 재현됩니다.
REF_YEAR, REF_MONTH = 2026, 10

# 신차대비가격이 이 값을 넘는 매물 = "중고가가 신차가보다 비싼" 특이 매물
# (인기 차종 프리미엄, 사이트 계산 오류 등). 점수를 따로 한 번 더 보여주기 위한 기준.
ANOMALY_CUT = 100

MIN_BRAND_COUNT = 8   # 이 건수 이상인 브랜드만 개별 특성으로 사용 (나머지는 '기타')
MIN_TOKEN_COUNT = 12  # 이름 단어가 이 건수 이상 등장해야 특성으로 사용

NUMERIC_BASE = [
    "주행거리", "배기량", "소유자변경횟수",
    "사고상세_전손", "사고상세_침수전손", "사고상세_침수분손", "사고상세_도난",
    "보험_내차피해(횟수)", "보험_내차피해(가격)",
    "사고상세_타차가해(횟수)", "보험_타차피해(가격)",
]


# ---------------------------------------------------------------------------
# 데이터 로딩/정리 (기존 로직 유지)
# ---------------------------------------------------------------------------
def _to_numeric_lenient(series: pd.Series) -> pd.Series:
    """'72%', '1,234' 처럼 숫자 외 문자가 섞여 있어도 숫자 부분만 뽑아 변환."""
    cleaned = (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.extract(r"(-?\d+\.?\d*)", expand=False)
    )
    return pd.to_numeric(cleaned, errors="coerce")


def load_and_clean(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # 리스/렌트 승계 매물은 가격이 인수금이라 신차대비가격 모델에서 제외
    # (예전 버전 CSV에는 매물유형 컬럼이 없음)
    if "매물유형" in df.columns:
        df = df[df["매물유형"].isna() | (df["매물유형"] == "일반")]

    plain_numeric_cols = ["연식"] + NUMERIC_BASE + ["가격", "신차가"]
    for c in [c for c in plain_numeric_cols if c in df.columns]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["신차대비가격"] = _to_numeric_lenient(df["신차대비가격"])

    # 사고/보험 이력: 사고 없음 = 빈 값이 대부분이므로 0으로 채움
    accident_cols = [c for c in NUMERIC_BASE if c not in ("주행거리", "배기량")]
    accident_cols = [c for c in accident_cols if c in df.columns]
    df[accident_cols] = df[accident_cols].fillna(0)

    # 옵션 컬럼(유/무 → 1/0): 비어 있으면 정보없음 → 0
    option_cols = [c for c in df.columns if c.startswith("옵션_")]
    df[option_cols] = df[option_cols].apply(pd.to_numeric, errors="coerce").fillna(0)

    core_cols = [c for c in ["가격", "신차대비가격", "연식", "주행거리", "배기량"] if c in df.columns]
    df = df.dropna(subset=core_cols)

    median_ratio = df["신차대비가격"].median()
    if median_ratio <= 3:
        df = df[(df["신차대비가격"] > 0) & (df["신차대비가격"] <= 3)]
        print(f"[진단] 신차대비가격을 소수(0~3) 스케일로 판단 (중앙값={median_ratio:.2f})")
    else:
        df = df[(df["신차대비가격"] > 0) & (df["신차대비가격"] <= 300)]
        print(f"[진단] 신차대비가격을 퍼센트(%) 스케일로 판단 (중앙값={median_ratio:.2f})")

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 특성 만들기
#   fit_feature_state()  : 데이터에서 "어떤 브랜드/단어를 특성으로 쓸지" 목록을 정함
#   transform_features() : 그 목록대로 특성 표를 만듦 (새 매물 예측에도 그대로 사용)
#   목록은 dict 로 저장되므로 joblib 파일을 다른 곳에서 불러와도 동작합니다.
# ---------------------------------------------------------------------------
def _age_months(yeonsik: pd.Series) -> pd.Series:
    """01 스크립트의 '연식' 숫자(= 12*(2022-년) + 월) 에서 년/월을 복원해
    기준 시점(REF_YEAR/REF_MONTH) 기준 나이(개월)로 변환한다."""
    k = yeonsik.astype(int)
    month = ((k - 1) % 12) + 1          # 1~12
    year = 2022 - (k - month) // 12
    return 12 * (REF_YEAR - year) + (REF_MONTH - month)


def _name_tokens(names: pd.Series) -> pd.Series:
    cleaned = names.astype(str).str.replace(r"[-/()]", " ", regex=True).str.lower()
    return cleaned.apply(lambda s: set(s.split()))


def fit_feature_state(df: pd.DataFrame) -> dict:
    brand_counts = df["브랜드"].value_counts()
    top_brands = [b for b, n in brand_counts.items() if n >= MIN_BRAND_COUNT]

    token_counts = {}
    for toks in _name_tokens(df["이름"]):
        for t in toks:
            token_counts[t] = token_counts.get(t, 0) + 1
    vocab = sorted(t for t, n in token_counts.items() if n >= MIN_TOKEN_COUNT)

    state = {
        "top_brands": top_brands,
        "vocab": vocab,
        "fuel_levels": sorted(df["연료"].astype(str).unique())[1:],  # 첫 값은 기준(drop_first)
        "option_cols": [c for c in df.columns if c.startswith("옵션_")],
    }
    state["columns"] = list(transform_features(df, state).columns)
    return state


def _one_hot(values: pd.Series, levels: list, prefix: str) -> pd.DataFrame:
    """values 가 각 level 과 같은지를 0/1 열로. 열을 하나씩 만들지 않고 한 번의 비교로 계산."""
    hits = values.to_numpy()[:, None] == np.asarray(levels, dtype=object)[None, :]
    return pd.DataFrame(hits.astype(int), index=values.index, columns=[f"{prefix}{lv}" for lv in levels])


def transform_features(df: pd.DataFrame, state: dict) -> pd.DataFrame:
    # [성능] 열 164개를 X[c] = ... 로 하나씩 추가하면 pandas 가 추가할 때마다 내부 메모리를
    # 다시 잡아서(PerformanceWarning: highly fragmented), 차 1대 예측에도 60ms 넘게 걸렸다.
    # 열을 모두 모아 두었다가 마지막에 한 번에 DataFrame 으로 만든다. 열 순서/값은 예전과 같다.
    def numeric(c):
        if c not in df.columns:
            return pd.Series(0.0, index=df.index)
        return pd.to_numeric(df[c], errors="coerce").fillna(0)

    base = {c: numeric(c) for c in NUMERIC_BASE}
    base["주행거리"] = base["주행거리"].clip(upper=1_000_000)  # 데이터 오류(1.8억 km 등) 방지
    for c in state["option_cols"]:
        base[c] = numeric(c)

    age = _age_months(df["연식"])
    base["나이개월"] = age
    base["log주행"] = np.log1p(base["주행거리"])
    base["연당주행"] = base["주행거리"] / np.maximum(age / 12, 0.5)

    # 신차가(옵션 미포함, 원): 차급/트림을 숫자 하나로 요약해 주는 피처.
    # 예측 시점에도 알 수 있는 값이라 누수가 아님(가격은 입력으로 쓰지 않음).
    # 없는 행(예전 CSV, 사이트 미제공)은 0 + '신차가있음'=0 으로 표시
    msrp = pd.to_numeric(df["신차가"], errors="coerce") if "신차가" in df.columns else pd.Series(np.nan, index=df.index)
    msrp = msrp.where(msrp > 0)
    base["신차가있음"] = msrp.notna().astype(int)
    base["log신차가"] = np.log(msrp).fillna(0)

    generation = pd.to_numeric(
        df["이름"].astype(str).str.extract(r"(\d+)세대", expand=False), errors="coerce"
    ).fillna(0)

    # 이름 토큰: 단어마다 전체 행을 다시 훑지 않고, 행마다 한 번만 훑어 해당 칸에 1을 표시
    vocab_index = {t: j for j, t in enumerate(state["vocab"])}
    tokens = np.zeros((len(df), len(vocab_index)), dtype=int)
    for i, toks in enumerate(_name_tokens(df["이름"])):
        for t in toks:
            j = vocab_index.get(t)
            if j is not None:
                tokens[i, j] = 1

    X = pd.concat([
        pd.DataFrame(base, index=df.index),
        _one_hot(df["연료"].astype(str), state["fuel_levels"], "연료_"),
        _one_hot(df["브랜드"], state["top_brands"], "브_"),
        pd.DataFrame({"세대": generation}, index=df.index),
        pd.DataFrame(tokens, index=df.index, columns=[f"토큰_{t}" for t in state["vocab"]]),
    ], axis=1)
    return X.reindex(columns=state.get("columns", X.columns), fill_value=0)


def build_baseline_features(df: pd.DataFrame) -> pd.DataFrame:
    """비교용: 기존 스크립트가 쓰던 특성 (원래 연식 숫자 + 연료/색상 원-핫)."""
    cols = ["주행거리", "연식", "배기량"] + [c for c in NUMERIC_BASE if c not in ("주행거리", "배기량")]
    cols += [c for c in df.columns if c.startswith("옵션_")]
    X = df[cols].copy()
    return pd.concat([X, pd.get_dummies(df[["연료", "색상"]].astype(str), drop_first=True)], axis=1)


# ---------------------------------------------------------------------------
# 평가
# ---------------------------------------------------------------------------
def make_gbr():
    """최종 모델. 04_retrain.py 도 같은 설정으로 챌린저를 학습한다."""
    return GradientBoostingRegressor(n_estimators=500, learning_rate=0.03, max_depth=3,
                                     subsample=0.8, loss="huber", alpha=0.9,
                                     random_state=RANDOM_STATE)


def listing_no(links: pd.Series) -> pd.Series:
    """매물 링크의 no= 번호. 등록 순서대로 커지므로 '언제 올라온 매물인지'의 대리 지표로 쓴다."""
    return pd.to_numeric(links.astype(str).str.extract(r"no=(\d+)", expand=False), errors="coerce")


def holdout_metrics(model, X_test, y_test):
    pred = model.predict(X_test)
    return {
        "r2": r2_score(y_test, pred),
        "mae": mean_absolute_error(y_test, pred),
        "rmse": np.sqrt(mean_squared_error(y_test, pred)),
    }


def repeated_cv(make_model, X, y):
    """5-fold 교차검증을 CV_REPEATS번 (매번 다르게 섞어서) 반복.
    반환: 전체 R², '신차대비 ANOMALY_CUT% 이하만' R², 전체 MAE, R² 표준편차"""
    y = np.asarray(y)
    r2_all, r2_norm, maes = [], [], []
    for rep in range(CV_REPEATS):
        oof = np.zeros(len(y))
        for tr, te in KFold(CV_SPLITS, shuffle=True, random_state=RANDOM_STATE + rep).split(X):
            m = make_model()
            m.fit(X.iloc[tr], y[tr])
            oof[te] = m.predict(X.iloc[te])
        normal = y <= ANOMALY_CUT
        r2_all.append(r2_score(y, oof))
        r2_norm.append(r2_score(y[normal], oof[normal]))
        maes.append(mean_absolute_error(y, oof))
    return {
        "cv_r2": float(np.mean(r2_all)), "cv_r2_std": float(np.std(r2_all)),
        "cv_r2_normal": float(np.mean(r2_norm)), "cv_mae": float(np.mean(maes)),
    }


def main():
    df = load_and_clean(CSV_PATH)
    print(f"전처리 후 관측치 수: {len(df)}건")
    if len(df) < 30:
        raise SystemExit("데이터가 너무 적어 모델을 만들 수 없습니다. cars_processed.csv를 확인하세요.")

    y = df[TARGET_COL].reset_index(drop=True)
    n_anom = int((y > ANOMALY_CUT).sum())
    print(f"신차대비가격 {ANOMALY_CUT}% 초과 매물: {n_anom}건 ({n_anom / len(y):.1%})")

    state = fit_feature_state(df)
    X = transform_features(df, state)
    X_base = build_baseline_features(df)
    print(f"특성 개수: 개선 {X.shape[1]}개 / 기존 {X_base.shape[1]}개")

    # 모델 정의 -------------------------------------------------------------
    def make_linear():  return make_pipeline(StandardScaler(), LinearRegression())
    def make_ridge():   return make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    def make_rf():
        return RandomForestRegressor(n_estimators=400, min_samples_leaf=2, max_features=0.5,
                                     random_state=RANDOM_STATE, n_jobs=-1)
    def make_rf_old():
        return RandomForestRegressor(n_estimators=300, min_samples_leaf=2,
                                     random_state=RANDOM_STATE, n_jobs=-1)

    print(f"신차가 있는 행: {int(X['신차가있음'].sum())}건 / {len(X)}건")
    X_no_msrp = X.drop(columns=["신차가있음", "log신차가"])

    candidates = [
        ("선형회귀",                make_linear, X),
        ("릿지회귀",                make_ridge,  X),
        ("랜덤포레스트",            make_rf,     X),
        ("GradientBoosting(huber)", make_gbr,    X),
        ("GBR (신차가 제외)",       make_gbr,    X_no_msrp),
        ("[기존] 랜덤포레스트",     make_rf_old, X_base),
    ]

    idx_train, idx_test = train_test_split(
        np.arange(len(df)), test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    rows = []
    for name, maker, feats in candidates:
        m = maker().fit(feats.iloc[idx_train], y.iloc[idx_train])
        hold = holdout_metrics(m, feats.iloc[idx_test], y.iloc[idx_test])
        cv = repeated_cv(maker, feats, y)
        rows.append({
            "모델": name,
            "홀드아웃 R²": hold["r2"],
            "교차검증 R²(전체)": cv["cv_r2"],
            "±": cv["cv_r2_std"],
            f"교차검증 R²({ANOMALY_CUT}%이하만)": cv["cv_r2_normal"],
            "교차검증 MAE": cv["cv_mae"],
        })
        print(f"  {name} 평가 완료")

    results_df = pd.DataFrame(rows)
    pd.options.display.float_format = "{:.3f}".format
    print("\n=== 모델 비교 ===")
    print(results_df.to_string(index=False))
    print(
        "\n읽는 법: '홀드아웃'은 한 번 나눈 결과라 운에 따라 ±0.05 흔들립니다. "
        "'교차검증(전체)'이 더 믿을 만한 대표 점수입니다.\n"
        f"'{ANOMALY_CUT}%이하만'은 신차보다 비싸게 팔리는 특이 매물({n_anom}건)을 평가에서 뺀 "
        "참고용 점수이므로, 보고할 때는 반드시 그 조건을 같이 밝혀야 합니다."
    )

    # 최종 모델: GradientBoosting ---------------------------------------------
    final_name = "GradientBoosting(huber)"
    holdout_model = make_gbr().fit(X.iloc[idx_train], y.iloc[idx_train])
    pred_test = holdout_model.predict(X.iloc[idx_test])
    y_test = y.iloc[idx_test].to_numpy()

    # 그래프: 홀드아웃 실제값 vs 예측값
    plt.figure(figsize=(6.5, 6.5))
    normal = y_test <= ANOMALY_CUT
    plt.scatter(y_test[normal], pred_test[normal], alpha=0.5, label="일반 매물")
    if (~normal).any():
        plt.scatter(y_test[~normal], pred_test[~normal], alpha=0.8, color="tab:orange",
                    label=f"{ANOMALY_CUT}% 초과 매물")
    lims = [min(y_test.min(), pred_test.min()), max(y_test.max(), pred_test.max())]
    plt.plot(lims, lims, "r--", label="완벽한 예측선")
    plt.xlabel(f"실제 {TARGET_COL}")
    plt.ylabel(f"예측 {TARGET_COL}")
    plt.title(f"{final_name}: 실제값 vs 예측값 (홀드아웃)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("prediction_actual_vs_pred.png", dpi=150)
    print("\n그래프 저장: prediction_actual_vs_pred.png")

    # 전체 데이터로 다시 학습해서 저장 (새 매물 예측용)
    # 랜덤포레스트는 API에서 쓰지 않고 파일 크기만 키워서(약 19MB) 저장하지 않음
    final_model = make_gbr().fit(X, y)
    lin_all = make_linear().fit(X, y)

    joblib.dump({
        "model": final_model,
        "linear_model": lin_all,
        "feature_state": state,
        "feature_columns": list(X.columns),
        "target_col": TARGET_COL,
        "ref_year_month": (REF_YEAR, REF_MONTH),
        # 재학습 파이프라인(04)이 '이 모델이 본 적 없는 새 매물'을 고르는 기준
        "trained_max_no": int(listing_no(df["링크"]).max()),
        "n_train": len(df),
    }, "price_model.joblib")
    print("모델 저장: price_model.joblib")

    importance_df = pd.DataFrame({
        "변수": X.columns, "중요도": final_model.feature_importances_,
    }).sort_values("중요도", ascending=False)
    print("\nGradientBoosting 변수 중요도 상위 15개:")
    print(importance_df.head(15).to_string(index=False))

    lin_coef = lin_all.named_steps["linearregression"].coef_
    coef_df = pd.DataFrame({"변수": X.columns, "표준화 계수": lin_coef}).sort_values(
        "표준화 계수", key=abs, ascending=False)

    importance_df.to_csv("feature_importance.csv", index=False, encoding="utf-8-sig")
    coef_df.to_csv("linear_coefficients.csv", index=False, encoding="utf-8-sig")
    print("\n변수 중요도/계수 저장: feature_importance.csv, linear_coefficients.csv")

    pool = build_recommend_pool(df, X, y, make_gbr)
    pool.to_csv(POOL_CSV, index=False, encoding="utf-8-sig")
    print(f"추천 후보 저장: {POOL_CSV} ({len(pool)}건, 시세보다 저렴 {int(pool['저렴'].sum())}건, "
          f"이상 매물 의심 {int(pool['이상매물의심'].sum())}건)")


def bargain_group(brands: pd.Series, age_months: pd.Series) -> pd.Series:
    """'저렴' 판정용 비교 그룹: 국산/수입 x 연식대. 예: '국산_6~10년'"""
    origin = np.where(brands.isin(DOMESTIC_BRANDS), "국산", "수입")
    labels = ["~3년", "3~6년", "6~10년", "10~15년", "15년~"]
    # 연식이 기준 시점보다 미래로 입력된 매물(나이 음수)도 어느 그룹엔가 들어가도록 범위를 자름
    years = (age_months / 12).clip(lower=0, upper=BARGAIN_AGE_BINS[-1] - 1)
    age = pd.cut(years, BARGAIN_AGE_BINS, labels=labels).astype(str)
    return pd.Series(origin, index=brands.index) + "_" + age


def build_recommend_pool(df: pd.DataFrame, X: pd.DataFrame, y: pd.Series, make_model) -> pd.DataFrame:
    """추천 후보(현재 판매 중인 매물)마다 적정 가격(시세)을 계산한다.

    후보 매물은 학습 데이터에 그대로 들어 있으므로, 전체 데이터로 학습한 모델로
    예측하면 정답을 외운 값이 나와 모든 매물이 '적정가'로 보인다. 그래서 각 매물은
    자기가 빠진 fold로 학습한 모델로 예측한다(out-of-fold).
    적정가격 = 예측 신차대비가격 x 신차가, 할인율 = (적정가격 - 판매가) / 적정가격

    '저렴' 판정과 추천 순서는 할인율이 아니라 '비슷한 차(그룹) 안에서의 순위'로 정한다.
    할인율은 예측 신차대비가격으로 나누기 때문에, 신차대비가 낮은 오래된 차일수록 같은
    예측 오차가 큰 할인율로 부풀려진다(예: 예측 15%에 오차 5%p = 할인율 33%). 그래서 할인율 순으로
    뽑으면 상위 추천의 80%가 10년 넘은 차였다(전체 매물 중 비율은 35%).
      잔차      = 실제 신차대비 - 예측 신차대비 (음수 = 예측보다 싸게 나옴)
      그룹      = 국산/수입 x 연식대 (BARGAIN_AGE_BINS)
      시세백분위 = 그룹 안에서 잔차가 몇 % 위치인지 (0에 가까울수록 비슷한 차 중 가장 쌈)
      저렴      = 잔차가 그룹의 하위 BARGAIN_QUANTILE(10%) 기준보다 낮음
    """
    oof = np.zeros(len(y))
    for tr, te in KFold(CV_SPLITS, shuffle=True, random_state=RANDOM_STATE).split(X):
        oof[te] = make_model().fit(X.iloc[tr], y.iloc[tr]).predict(X.iloc[te])

    residual = pd.Series(np.asarray(y) - oof, index=df.index)
    group = bargain_group(df["브랜드"], _age_months(df["연식"]))
    small = group.map(group.value_counts()) < MIN_GROUP_SIZE
    group = group.where(~small, "전체")  # 건수가 적은 그룹은 기준이 흔들려서 전체 기준을 씀
    cut_by_group = residual.groupby(group).quantile(BARGAIN_QUANTILE)
    cut_by_group["전체"] = residual.quantile(BARGAIN_QUANTILE)
    cut = group.map(cut_by_group)
    percentile = residual.groupby(group).rank(pct=True)

    on_sale = df["신차가"].notna() & (df["가격"] > 0)
    if "가격상태" in df.columns:
        on_sale &= df["가격상태"] == "판매중"
    if "수집일" in df.columns:  # 마지막 크롤링에 없던 매물은 이미 팔렸을 수 있음
        on_sale &= df["수집일"] == df["수집일"].max()

    k = df["연식"].astype(int)
    month = ((k - 1) % 12) + 1
    pool = pd.DataFrame({
        "이름": df["이름"].astype(str).str.split().str.join(" "),
        "브랜드": df["브랜드"],
        "연식_년": 2022 - (k - month) // 12,
        "연식_월": month,
        "주행거리": df["주행거리"],
        "연료": df["연료"].astype(str).replace({"0": "디젤", "1": "LPG", "2": "가솔린"}),
        "배기량": df["배기량"],
        "보험이력등록": pd.to_numeric(df["보험이력등록"], errors="coerce").fillna(0).astype(int),
        "사고횟수": df["보험_내차피해(횟수)"],
        "전손침수": df[["사고상세_전손", "사고상세_침수전손", "사고상세_침수분손"]].sum(axis=1),
        "가격": df["가격"],
        "신차가": df["신차가"],
        "신차대비가격": y,
        "예측_신차대비가격": oof,
        "링크": df["링크"],
    })
    pool["적정가격"] = (pool["예측_신차대비가격"] / 100 * pool["신차가"]).round(-4)
    pool["시세차이"] = pool["적정가격"] - pool["가격"]
    pool["할인율"] = pool["시세차이"] / pool["적정가격"]
    pool["비교그룹"] = group.to_numpy()
    pool["시세백분위"] = percentile.to_numpy()
    # 이 가격 이하면 비슷한 차 중 하위 10% = '저렴'. 사이트의 신차대비가격은 정수로 반올림된 값이라,
    # 화면에 보이는 기준 가격과 판정이 어긋나지 않도록 판정은 원 단위 가격으로 한다.
    threshold = ((oof + cut.to_numpy()) / 100 * pool["신차가"]).clip(lower=0)
    pool["저렴기준가격"] = threshold.round(-4)
    pool["저렴"] = pool["가격"] <= threshold
    # 시세의 절반 이하 가격은 대부분 진짜 급매가 아니라 데이터 오류
    # (계약금만 적은 미끼 매물, 사이트 신차가 오류, 연식 오입력 등) -> 추천에서 제외
    pool["이상매물의심"] = (pool["할인율"] > SUSPECT_DISCOUNT) | (pool["신차대비가격"] < 5)
    return pool[on_sale.to_numpy()].reset_index(drop=True)


def predict_new(new_cars: pd.DataFrame, model_path: str = "price_model.joblib") -> np.ndarray:
    """새 매물의 신차대비가격(%)을 예측한다.

    new_cars 는 cars_processed.csv 와 같은 컬럼 이름을 가져야 한다.
    최소한 '이름', '브랜드', '연식', '주행거리', '연료', '배기량' 이 필요하며
    (연식은 01 스크립트와 같은 숫자 형식: 12*(2022-년)+월), 나머지는 없으면 0으로 처리된다.

    사용 예:
        import pandas as pd
        from importlib import import_module   # 파일명이 숫자로 시작하므로 이렇게 불러오거나
        # 이 파일 아래쪽에서 직접 호출해도 된다
        new = pd.DataFrame([{"이름": "현대 아반떼 1.6", "브랜드": "현대", "연식": 12*(2022-2021)+5,
                             "주행거리": 50000, "연료": 2, "배기량": 1598}])
        print(predict_new(new))
    """
    saved = joblib.load(model_path)
    X_new = transform_features(new_cars, saved["feature_state"])
    return saved["model"].predict(X_new)


if __name__ == "__main__":
    main()