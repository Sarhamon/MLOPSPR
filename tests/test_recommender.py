import pandas as pd
import pytest

from recommender import recommend


def listing(no, 가격, 적정가격, 백분위, **kw):
    row = {
        "이름": f"차{no}", "브랜드": "현대", "연식_년": 2020, "주행거리": 50_000, "연료": "가솔린",
        "보험이력등록": 1, "사고횟수": 0, "전손침수": 0,
        "가격": 가격, "적정가격": 적정가격, "시세백분위": 백분위, "저렴기준가격": 적정가격 * 0.85,
        "링크": f"https://www.bobaedream.co.kr/mycar/mycar_view.php?no={no}&gubun=K",
    }
    row.update(kw)
    row["시세차이"] = row["적정가격"] - row["가격"]
    row["할인율"] = row["시세차이"] / row["적정가격"]
    row["저렴"] = row["시세백분위"] <= 0.1
    row["이상매물의심"] = row["할인율"] > 0.5
    return row


@pytest.fixture
def pool():
    return pd.DataFrame([
        listing(1, 10_000_000, 12_000_000, 0.08),                                  # 저렴
        listing(2, 15_000_000, 20_000_000, 0.05, 브랜드="기아"),                    # 저렴, 그룹 내 가장 쌈
        listing(3, 8_000_000, 8_500_000, 0.40, 연식_년=2015),                       # 시세 수준
        listing(4, 1_000_000, 50_000_000, 0.01),                                    # 이상 매물 의심
        listing(5, 30_000_000, 40_000_000, 0.02),                                   # 예산 초과
        listing(6, 9_000_000, 12_000_000, 0.09, 보험이력등록=0),
        listing(7, 9_500_000, 14_000_000, 0.30, 사고횟수=2, 연료="디젤", 연식_년=2010),  # 할인율은 가장 큼
    ])


def test_budget_and_suspect_filter(pool):
    names = [c["이름"] for c in recommend(pool, max_budget=20_000_000)]
    assert "차5" not in names  # 예산 초과
    assert "차4" not in names  # 적정가의 절반 이하 = 이상 매물 의심
    assert all(c["가격"] <= 20_000_000 for c in recommend(pool, max_budget=20_000_000))


def test_sorted_by_group_percentile_not_discount(pool):
    """할인율이 가장 큰 차7(오래된 차)보다, 비슷한 차 중 더 싸게 나온 차가 먼저 와야 한다."""
    cars = recommend(pool, max_budget=20_000_000)
    assert [c["이름"] for c in cars] == ["차2", "차1", "차6", "차7", "차3"]
    assert max(cars, key=lambda c: c["할인율"])["이름"] == "차7"


def test_conditions(pool):
    assert {c["이름"] for c in recommend(pool, 20_000_000, brand="기아")} == {"차2"}
    assert "차3" not in {c["이름"] for c in recommend(pool, 20_000_000, min_year=2018)}
    assert {c["이름"] for c in recommend(pool, 20_000_000, fuel="디젤")} == {"차7"}
    # 무사고만: 보험이력 미공개(차6), 사고 있음(차7) 제외
    assert {c["이름"] for c in recommend(pool, 20_000_000, no_accident=True)} == {"차1", "차2", "차3"}
    assert len(recommend(pool, 20_000_000, top_n=2)) == 2
    assert recommend(pool, 500_000) == []


def test_reason_text(pool):
    by_name = {c["이름"]: c for c in recommend(pool, 20_000_000, top_n=10)}
    assert "시세보다 200만원 저렴" in by_name["차1"]["추천이유"]
    assert "비슷한 차 중 하위 8% 가격" in by_name["차1"]["추천이유"]
    assert "시세 수준" in by_name["차3"]["추천이유"] and "722만원 이하면 저렴" in by_name["차3"]["추천이유"]
    assert "보험이력 미공개" in by_name["차6"]["추천이유"]
    assert "보험사고 2회" in by_name["차7"]["추천이유"]
