import pandas as pd
import pytest

from recommender import recommend


def listing(no, 가격, 적정가격, **kw):
    row = {
        "이름": f"차{no}", "브랜드": "현대", "연식_년": 2020, "주행거리": 50_000, "연료": "가솔린",
        "보험이력등록": 1, "사고횟수": 0, "전손침수": 0,
        "가격": 가격, "적정가격": 적정가격, "오차범위": 1_000_000,
        "링크": f"https://www.bobaedream.co.kr/mycar/mycar_view.php?no={no}&gubun=K",
    }
    row.update(kw)
    row["시세차이"] = row["적정가격"] - row["가격"]
    row["할인율"] = row["시세차이"] / row["적정가격"]
    row["저렴"] = row["시세차이"] > row["오차범위"]
    row["이상매물의심"] = row["할인율"] > 0.5
    return row


@pytest.fixture
def pool():
    return pd.DataFrame([
        listing(1, 10_000_000, 12_000_000),                        # 할인율 0.167
        listing(2, 15_000_000, 20_000_000, 브랜드="기아"),          # 0.25
        listing(3, 8_000_000, 8_500_000, 연식_년=2015),             # 0.059, 시세 수준
        listing(4, 1_000_000, 50_000_000),                          # 0.98 -> 이상 매물
        listing(5, 30_000_000, 40_000_000),                         # 예산 초과
        listing(6, 9_000_000, 12_000_000, 보험이력등록=0),          # 0.25
        listing(7, 9_500_000, 13_000_000, 사고횟수=2, 연료="디젤"),  # 0.269
    ])


def test_budget_and_suspect_filter(pool):
    names = [c["이름"] for c in recommend(pool, max_budget=20_000_000)]
    assert "차5" not in names  # 예산 초과
    assert "차4" not in names  # 적정가의 절반 이하 = 이상 매물 의심
    assert all(c["가격"] <= 20_000_000 for c in recommend(pool, max_budget=20_000_000))


def test_sorted_by_discount_rate(pool):
    cars = recommend(pool, max_budget=20_000_000)
    rates = [c["할인율"] for c in cars]
    assert rates == sorted(rates, reverse=True)
    assert cars[0]["이름"] == "차7"


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
    assert "시세 수준" in by_name["차3"]["추천이유"] and not by_name["차3"]["저렴"]
    assert "보험이력 미공개" in by_name["차6"]["추천이유"]
    assert "보험사고 2회" in by_name["차7"]["추천이유"]
