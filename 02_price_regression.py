"""
중고차 가격 예측 모델 (파이썬 버전, R 불필요)

02_price_regression.R을 대체합니다. R/RStudio 설치 없이 파이썬 환경만으로
동작하며, scikit-learn으로 선형회귀 + 랜덤포레스트 두 가지 모델을 만들어
비교합니다.

실행 순서:
    1) 01_crawl_and_preprocess.py 를 먼저 돌려 cars_processed.csv 생성
    2) pip install scikit-learn joblib matplotlib
    3) python3 02_price_prediction.py
"""

import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # 화면 없는 환경(서버 등)에서도 그래프 파일 저장 가능
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# 설정
# ---------------------------------------------------------------------------
CSV_PATH = "cars_processed.csv"

# 무엇을 예측할지: "신차대비가격"(추천 - 차종/연식 영향을 어느 정도 정규화한
# 비율값이라 여러 차종이 섞인 데이터에서 더 안정적) 또는 "가격"(원 단위 실제
# 거래가) 중 선택. 둘 다 궁금하면 스크립트를 두 번 돌리거나 아래에서 바꾸세요.
TARGET_COL = "신차대비가격"

RANDOM_STATE = 42
TEST_SIZE = 0.2


def _to_numeric_lenient(series: pd.Series) -> pd.Series:
    """'72%', '72 %', '72.5%', '1,234' 처럼 숫자 외 문자가 섞여 있어도
    숫자 부분만 뽑아 변환한다. pd.to_numeric은 '%'가 붙으면 통째로 NaN
    처리해버려서, 신차대비가격처럼 퍼센트 표기가 섞인 컬럼에 그대로 쓰면
    거의 모든 값이 NaN이 되는 문제가 있었다(실제로 591건 중 3건만 남았음).
    """
    cleaned = (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.extract(r"(-?\d+\.?\d*)", expand=False)
    )
    return pd.to_numeric(cleaned, errors="coerce")


def load_and_clean(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    plain_numeric_cols = [
        "연식", "주행거리", "배기량", "소유자변경횟수",
        "사고상세_전손", "사고상세_침수전손", "사고상세_침수분손", "사고상세_도난",
        "보험_내차피해(횟수)", "보험_내차피해(가격)",
        "사고상세_타차가해(횟수)", "보험_타차피해(가격)",
        "가격",
    ]
    plain_numeric_cols = [c for c in plain_numeric_cols if c in df.columns]
    for c in plain_numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # [수정] 신차대비가격은 '72%' 처럼 % 기호가 붙어 있을 수 있어 별도의
    # 관대한 파서로 숫자만 추출한다.
    if "신차대비가격" in df.columns:
        df["신차대비가격"] = _to_numeric_lenient(df["신차대비가격"])

    # 사고/보험 이력 컬럼: 사고 없음 = 원래 빈 값(NA)인 경우가 대부분이므로
    # 행을 버리지 않고 0(사고 없음/0건/0원)으로 채운다.
    accident_cols = [
        "소유자변경횟수", "사고상세_전손", "사고상세_침수전손", "사고상세_침수분손", "사고상세_도난",
        "보험_내차피해(횟수)", "보험_내차피해(가격)", "사고상세_타차가해(횟수)", "보험_타차피해(가격)",
    ]
    accident_cols = [c for c in accident_cols if c in df.columns]
    df[accident_cols] = df[accident_cols].fillna(0)

    # 옵션 컬럼(유/무 → 1/0)도 비어 있으면 "정보없음"이므로 0으로 채운다.
    option_cols = [c for c in df.columns if c.startswith("옵션_")]
    df[option_cols] = df[option_cols].apply(pd.to_numeric, errors="coerce").fillna(0)

    # 핵심 컬럼(가격/신차대비가격/연식/주행거리/배기량)은 반드시 있어야 함
    core_cols = [c for c in ["가격", "신차대비가격", "연식", "주행거리", "배기량"] if c in df.columns]
    df = df.dropna(subset=core_cols)

    # [수정] 신차대비가격이 소수(0~3) 스케일이라고 잘못 가정하고
    # "0 < x <= 3"으로 필터링했던 게 버그였습니다. 실제로는 퍼센트 스케일
    # (예: 72)로 보입니다. 스케일을 미리 단정하지 않고, 남아 있는 값의
    # 중앙값(median)을 보고 그 자리에서 판단합니다.
    if "신차대비가격" in df.columns and len(df) > 0:
        median_ratio = df["신차대비가격"].median()
        if median_ratio <= 3:
            # 0~1(또는 0~3) 사이 소수 스케일로 보임
            df = df[(df["신차대비가격"] > 0) & (df["신차대비가격"] <= 3)]
            print(f"[진단] 신차대비가격을 소수(0~3) 스케일로 판단 (중앙값={median_ratio:.2f})")
        else:
            # 퍼센트(0~100대, 드물게 그 이상) 스케일로 보임
            df = df[(df["신차대비가격"] > 0) & (df["신차대비가격"] <= 300)]
            print(f"[진단] 신차대비가격을 퍼센트(%) 스케일로 판단 (중앙값={median_ratio:.2f})")

    return df


def build_features(df: pd.DataFrame, target_col: str):
    """수치형 + 원-핫 인코딩된 범주형 특성 행렬(X)과 타깃(y)을 만든다."""
    feature_cols_numeric = [
        c for c in [
            "주행거리", "연식", "배기량", "소유자변경횟수",
            "사고상세_전손", "사고상세_침수전손", "사고상세_침수분손", "사고상세_도난",
            "보험_내차피해(횟수)", "보험_내차피해(가격)",
            "사고상세_타차가해(횟수)", "보험_타차피해(가격)",
        ] if c in df.columns
    ]
    option_cols = [c for c in df.columns if c.startswith("옵션_")]
    feature_cols_numeric += option_cols

    categorical_cols = [c for c in ["연료", "색상"] if c in df.columns]

    X_numeric = df[feature_cols_numeric].copy()
    X_categorical = pd.get_dummies(df[categorical_cols].astype(str), drop_first=True) if categorical_cols else pd.DataFrame(index=df.index)

    X = pd.concat([X_numeric, X_categorical], axis=1)
    y = df[target_col]

    return X, y


def evaluate(name, model, X_test, y_test):
    pred = model.predict(X_test)
    r2 = r2_score(y_test, pred)
    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    print(f"[{name}] R^2={r2:.3f}  MAE={mae:.4f}  RMSE={rmse:.4f}")
    return {"model": name, "r2": r2, "mae": mae, "rmse": rmse}


def main():
    df = load_and_clean(CSV_PATH)
    print(f"전처리 후 관측치 수: {len(df)}건")
    if len(df) < 10:
        raise SystemExit(
            f"전처리 후 남은 데이터가 {len(df)}건뿐이라 모델을 만들 수 없습니다. "
            "cars_processed.csv의 '신차대비가격' 컬럼 값을 직접 확인해보세요 "
            "(예: print(pd.read_csv('cars_processed.csv')['신차대비가격'].head(20)))."
        )
    if len(df) < 30:
        print("경고: 데이터가 너무 적어 모델이 불안정할 수 있습니다. "
              "크롤링 범위를 더 넓히는 걸 권장합니다.")

    X, y = build_features(df, TARGET_COL)
    print(f"특성 개수: {X.shape[1]}개 (수치형 + 원-핫 인코딩)")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    results = []

    # 1) 선형회귀 (해석하기 쉬움: 계수가 "이 옵션이 있으면 가격비율이 얼마나
    #    바뀌는지"를 그대로 보여줌)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    lin_model = LinearRegression()
    lin_model.fit(X_train_scaled, y_train)
    results.append(evaluate("선형회귀", lin_model, X_test_scaled, y_test))

    coef_df = pd.DataFrame({
        "변수": X.columns,
        "표준화 계수": lin_model.coef_,
    }).sort_values("표준화 계수", key=abs, ascending=False)
    print("\n선형회귀 계수 (절댓값 기준 영향력 순, 표준화된 값이라 서로 비교 가능):")
    print(coef_df.head(15).to_string(index=False))

    # 2) 릿지회귀 (변수가 많고 서로 상관돼 있을 때 선형회귀보다 안정적)
    ridge_model = Ridge(alpha=1.0)
    ridge_model.fit(X_train_scaled, y_train)
    results.append(evaluate("릿지회귀", ridge_model, X_test_scaled, y_test))

    # 3) 랜덤포레스트 (비선형 관계·변수 간 상호작용까지 잡아냄, 보통 더 정확함)
    rf_model = RandomForestRegressor(
        n_estimators=300, max_depth=None, min_samples_leaf=2,
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    rf_model.fit(X_train, y_train)  # 트리 모델은 스케일링 불필요
    results.append(evaluate("랜덤포레스트", rf_model, X_test, y_test))

    importance_df = pd.DataFrame({
        "변수": X.columns,
        "중요도": rf_model.feature_importances_,
    }).sort_values("중요도", ascending=False)
    print("\n랜덤포레스트 변수 중요도 상위 15개:")
    print(importance_df.head(15).to_string(index=False))

    # 교차검증으로 랜덤포레스트 성능을 한 번 더 확인 (train/test 한 번
    # 나눈 것만으로는 우연히 잘 나올 수 있어서 교차검증으로 보강)
    # [수정] cv=5를 고정해두면 데이터가 5건보다 적을 때(또는 각 fold가
    # 너무 작을 때) "n_splits=5 > n_samples" 오류로 죽습니다. 데이터
    # 건수에 맞춰 fold 수를 최대 5까지 자동으로 낮춥니다.
    n_splits = min(5, len(X))
    if n_splits < 2:
        print("\n데이터가 너무 적어 교차검증을 건너뜁니다 (최소 2건 필요).")
    else:
        cv_scores = cross_val_score(rf_model, X, y, cv=n_splits, scoring="r2")
        print(f"\n랜덤포레스트 {n_splits}-fold 교차검증 R^2: 평균 {cv_scores.mean():.3f} "
              f"(개별: {np.round(cv_scores, 3).tolist()})")

    # ---------------------------------------------------------------------
    # 결과 요약 및 저장
    # ---------------------------------------------------------------------
    results_df = pd.DataFrame(results)
    print("\n=== 모델 비교 ===")
    print(results_df.to_string(index=False))

    best_row = results_df.loc[results_df["r2"].idxmax()]
    print(f"\n가장 성능이 좋은 모델: {best_row['model']} (R^2={best_row['r2']:.3f})")

    # 실제값 vs 예측값 산점도 (랜덤포레스트 기준)
    rf_pred = rf_model.predict(X_test)
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, rf_pred, alpha=0.5)
    lims = [min(y_test.min(), rf_pred.min()), max(y_test.max(), rf_pred.max())]
    plt.plot(lims, lims, "r--", label="완벽한 예측선")
    plt.xlabel(f"실제 {TARGET_COL}")
    plt.ylabel(f"예측 {TARGET_COL}")
    plt.title("랜덤포레스트: 실제값 vs 예측값")
    plt.legend()
    plt.tight_layout()
    plt.savefig("prediction_actual_vs_pred.png", dpi=150)
    print("\n그래프 저장: prediction_actual_vs_pred.png")

    # 모델과 컬럼 목록, 스케일러를 저장해두면 나중에 새 매물 가격을 바로 예측 가능
    joblib.dump({
        "rf_model": rf_model,
        "linear_model": lin_model,
        "scaler": scaler,
        "feature_columns": list(X.columns),
        "target_col": TARGET_COL,
    }, "price_model.joblib")
    print("모델 저장: price_model.joblib")

    importance_df.to_csv("feature_importance.csv", index=False, encoding="utf-8-sig")
    coef_df.to_csv("linear_coefficients.csv", index=False, encoding="utf-8-sig")
    print("변수 중요도/계수 저장: feature_importance.csv, linear_coefficients.csv")


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# 새 매물 가격을 예측하고 싶을 때 (예시)
# ---------------------------------------------------------------------------
# import joblib
# import pandas as pd
#
# saved = joblib.load("price_model.joblib")
# rf_model = saved["rf_model"]
# feature_columns = saved["feature_columns"]
#
# new_car = pd.DataFrame([{
#     "주행거리": 50000, "연식": 24, "배기량": 2000,
#     # ... 나머지 특성도 cars_processed.csv 컬럼과 동일한 이름으로 채우기
# }])
# new_car = new_car.reindex(columns=feature_columns, fill_value=0)
# predicted = rf_model.predict(new_car)
# print(predicted)