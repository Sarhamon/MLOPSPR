"""04_retrain.py 의 데이터 검증: 정상 크롤링은 통과, 깨진 크롤링은 막아야 한다."""

from datetime import date, timedelta

import pandas as pd
import pytest


def crawl_csv(path, n=2000, day=None, ratio_share=0.3, price_share=0.5, n_brands=25):
    day = day or date.today().isoformat()
    rows = []
    for i in range(n):
        rows.append({
            "이름": f"브랜드{i % n_brands} 차", "브랜드": f"브랜드{i % n_brands}", "매물유형": "일반",
            "연식": 51, "주행거리": 50_000, "배기량": 1598,
            "신차대비가격": 45 if i < n * ratio_share else None,
            "가격": 28_900_000 if i < n * price_share else None,
            "수집일": day,
        })
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
    return str(path)


def test_normal_crawl_passes(retrain, tmp_path):
    problems, stats = retrain.validate_data(crawl_csv(tmp_path / "c.csv"))
    assert problems == []
    assert stats["rows"] == 2000


@pytest.mark.parametrize("kwargs, expected", [
    ({"day": (date.today() - timedelta(days=5)).isoformat()}, "오래됨"),
    ({"n": 500}, "매물 수"),
    ({"ratio_share": 0.05}, "신차대비가격 있는 비율"),
    ({"price_share": 0.1}, "가격 있는 비율"),
    ({"n_brands": 5}, "브랜드 수"),
])
def test_broken_crawl_is_blocked(retrain, tmp_path, kwargs, expected):
    problems, _ = retrain.validate_data(crawl_csv(tmp_path / "c.csv", **kwargs))
    assert any(expected in p for p in problems), problems


def test_only_latest_crawl_is_checked(retrain, tmp_path):
    """예전 수집분이 많이 쌓여 있어도, 마지막 크롤링이 깨졌으면 막아야 한다."""
    old = pd.read_csv(crawl_csv(tmp_path / "old.csv", day=(date.today() - timedelta(days=3)).isoformat()))
    new = pd.read_csv(crawl_csv(tmp_path / "new.csv", n=300))
    path = tmp_path / "c.csv"
    pd.concat([old, new]).to_csv(path, index=False, encoding="utf-8-sig")
    problems, stats = retrain.validate_data(str(path))
    assert stats["rows"] == 300 and problems
