"""저장소에 들어 있는 price_model.joblib / recommend_pool.csv 로 API 를 실제로 띄워 확인한다."""

import pytest
from fastapi.testclient import TestClient

AVANTE = {"이름": "현대 더 뉴 아반떼AD 1.6 스타일", "브랜드": "현대", "연식_년": 2018, "연식_월": 3,
          "주행거리": 68658, "연료": "가솔린", "배기량": 1598}


@pytest.fixture(scope="module")
def client():
    import app
    return TestClient(app.app)


def test_index_and_health(client):
    assert "추천 받기" in client.get("/").text
    health = client.get("/health").json()
    assert len(health["model_version"]) == 12 and health["pool_rows"] > 0


def test_predict_uses_training_features(client, pr):
    """학습-서빙 불일치 버그 재발 방지. 예전 API는 입력을 reindex 만 해서 나이개월/브랜드/이름 토큰 등이
    0으로 들어갔다. API 예측값이 학습 스크립트의 predict_new() 와 정확히 같아야 한다."""
    import app
    row = app.to_training_row(app.CarInput(**AVANTE))
    expected = pr.predict_new(row, str(app.MODEL_PATH))[0]
    assert client.post("/predict", json=AVANTE).json()["prediction"] == pytest.approx(expected)


def test_predict_depends_on_age(client):
    """주행거리가 같아도 연식이 새로우면 신차대비가격이 확실히 높아야 한다 (나이 피처가 빠지면 같아짐)."""
    old = client.post("/predict", json=AVANTE).json()["prediction"]
    new = client.post("/predict", json={**AVANTE, "연식_년": 2024}).json()["prediction"]
    assert 0 < old < new - 10 < 150


def test_recommend(client):
    res = client.post("/recommend", json={"최대예산": 20_000_000, "개수": 5}).json()
    assert res["request_id"] and res["count"] == len(res["cars"]) <= 5
    assert all(c["가격"] <= 20_000_000 for c in res["cars"])


def test_invalid_input_rejected(client):
    assert client.post("/recommend", json={"최대예산": -1}).status_code == 422
    assert client.post("/predict", json={"이름": "x"}).status_code == 422
    assert client.post("/click", json={"request_id": "x", "링크": "y", "rank": 0}).status_code == 422


def test_click(client):
    assert client.post("/click", json={"request_id": "abc", "링크": "https://x", "rank": 1}).status_code == 204
