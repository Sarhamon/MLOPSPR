import pandas as pd


def car(**kw):
    base = {"이름": "현대 더 뉴 아반떼AD 1.6 스타일", "브랜드": "현대", "연식": 12 * (2022 - 2018) + 3,
            "주행거리": 68658, "연료": "2", "배기량": 1598}
    return {**base, **kw}


def test_age_months_decodes_crawler_year(pr):
    """01 의 연식 숫자(12*(2022-년)+월)를 기준 시점(2026-10) 나이로 바꾼다."""
    yeonsik = pd.Series([12 * (2022 - 2018) + 3, 12 * (2022 - 2024) + 1, 12 * (2022 - 2026) + 10])
    assert pr._age_months(yeonsik).tolist() == [103, 33, 0]


def test_listing_no(pr):
    links = pd.Series(["https://www.bobaedream.co.kr/mycar/mycar_view.php?no=2265922&gubun=K", "링크없음"])
    nos = pr.listing_no(links)
    assert nos.iloc[0] == 2265922
    assert pd.isna(nos.iloc[1])


def test_transform_matches_training_columns(pr):
    """학습-서빙 불일치 버그 재발 방지: 새 매물도 학습 때와 같은 컬럼/순서로 만들어져야 한다."""
    train = pd.DataFrame([car(), car(브랜드="기아", 이름="기아 K5 2.0"), car(신차가=17_280_000)] * 10)
    state = pr.fit_feature_state(train)

    new = pd.DataFrame([car(브랜드="처음보는브랜드", 이름="처음보는 차")])
    X = pr.transform_features(new, state)
    assert list(X.columns) == state["columns"]
    assert X.filter(like="브_").sum(axis=1).iloc[0] == 0
    assert X["나이개월"].iloc[0] == 103
    assert X["신차가있음"].iloc[0] == 0 and X["log신차가"].iloc[0] == 0


def test_msrp_feature(pr):
    df = pd.DataFrame([car(신차가=17_280_000), car(신차가=None), car(신차가=0)])
    X = pr.transform_features(df, pr.fit_feature_state(df))
    assert X["신차가있음"].tolist() == [1, 0, 0]
    assert X["log신차가"].iloc[0] > 16


def test_load_and_clean_drops_lease_and_junk(pr, tmp_path):
    rows = [
        car(가격=11_300_000, 신차대비가격="65", 매물유형="일반"),
        car(가격=12_000_000, 신차대비가격="40", 매물유형="일반"),
        car(가격=2_000_000, 신차대비가격="소유", 매물유형="승계(만기 후 소유)"),  # 예전 버그 데이터
        car(가격=3_000_000, 신차대비가격="50", 매물유형="승계(리스종류 운용)"),
        car(가격=None, 신차대비가격=None, 매물유형="일반"),
    ]
    path = tmp_path / "cars.csv"
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
    df = pr.load_and_clean(str(path))
    assert df["신차대비가격"].tolist() == [65, 40]
