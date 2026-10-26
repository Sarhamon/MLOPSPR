"""
예산 기반 중고차 추천

02_price_regression.py 가 만든 recommend_pool.csv(현재 판매 중인 매물 + 모델이
계산한 적정 가격)에서, 예산 안에 드는 매물을 시세 대비 가장 싸게 나온 순서로 고른다.
순서는 '비슷한 차(국산/수입 x 연식대) 안에서 얼마나 싸게 나왔는지'(시세백분위)로 정한다.
할인율로 정렬하면 신차대비가 낮은 오래된 차의 예측 오차가 큰 할인율로 부풀려져
상위가 오래된 차로 쏠렸기 때문이다 (build_recommend_pool 설명 참고).

사용 예:
    from recommender import load_pool, recommend
    print(recommend(load_pool(), max_budget=20_000_000, top_n=5))
"""

from pathlib import Path

import pandas as pd

POOL_PATH = Path(__file__).resolve().parent / "recommend_pool.csv"


def load_pool(path: Path = POOL_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def _reason(car: pd.Series) -> str:
    parts = []
    if car["저렴"]:
        parts.append(f"시세보다 {car['시세차이'] / 10000:,.0f}만원 저렴 (적정가 {car['적정가격'] / 10000:,.0f}만원, "
                     f"비슷한 차 중 하위 {max(car['시세백분위'] * 100, 1):.0f}% 가격)")
    else:
        parts.append(f"시세 수준 (적정가 {car['적정가격'] / 10000:,.0f}만원, "
                     f"{car['저렴기준가격'] / 10000:,.0f}만원 이하면 저렴)")
    parts.append(f"{car['연식_년']}년식 {car['주행거리'] / 10000:.1f}만km")
    if car["보험이력등록"] == 0:
        parts.append("보험이력 미공개")
    elif car["사고횟수"] == 0 and car["전손침수"] == 0:
        parts.append("보험사고 없음")
    else:
        parts.append(f"보험사고 {car['사고횟수']:.0f}회")
    return ", ".join(parts)


def recommend(
    pool: pd.DataFrame,
    max_budget: float,
    top_n: int = 10,
    brand: str | None = None,
    min_year: int | None = None,
    max_km: float | None = None,
    fuel: str | None = None,
    no_accident: bool = False,
) -> list[dict]:
    """예산(원)과 조건에 맞는 매물을 비슷한 차 대비 싼 순서(시세백분위 낮은 순)로 top_n개 돌려준다."""
    c = pool[(pool["가격"] <= max_budget) & ~pool["이상매물의심"]]
    if brand:
        c = c[c["브랜드"] == brand]
    if min_year:
        c = c[c["연식_년"] >= min_year]
    if max_km:
        c = c[c["주행거리"] <= max_km]
    if fuel:
        c = c[c["연료"] == fuel]
    if no_accident:  # 보험이력이 공개되고 사고/전손/침수가 0인 차만
        c = c[(c["보험이력등록"] == 1) & (c["사고횟수"] == 0) & (c["전손침수"] == 0)]

    c = c.sort_values(["시세백분위", "할인율"], ascending=[True, False]).head(top_n)
    return [
        {
            "이름": car["이름"],
            "가격": int(car["가격"]),
            "적정가격": int(car["적정가격"]),
            "할인율": round(float(car["할인율"]), 3),
            "시세백분위": round(float(car["시세백분위"]), 3),
            "저렴": bool(car["저렴"]),
            "추천이유": _reason(car),
            "링크": car["링크"],
        }
        for _, car in c.iterrows()
    ]
