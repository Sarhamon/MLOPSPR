"""운영 성능('저렴' 추천 판매율 - '시세 수준' 추천 판매율) 계산.

추천은 모델 버전이 아니라 매일 다시 만드는 추천 후보로 정해지므로, 모델 버전으로 거르지 않고
최근 LIFT_WINDOW_DAYS 일의 확정 정답으로 계산한다.
"""

from datetime import date

import pandas as pd


def write_labels(retrain, tmp_path, monkeypatch, rows):
    monkeypatch.chdir(tmp_path)
    retrain.LABELS_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["model_version", "추천일", "저렴", "판매여부"]).to_csv(retrain.LABELS_CSV, index=False)


def test_lift_ignores_model_version_and_old_labels(retrain, tmp_path, monkeypatch):
    n = retrain.MIN_LABELS
    rows = (
        [("v1", "2026-10-30", True, "판매됨")] * n            # 저렴: 모두 팔림 (모델 버전이 섞여 있어도 함께 계산)
        + [("v2", "2026-10-31", False, "미판매")] * n         # 시세 수준: 모두 안 팔림
        + [("v2", "2026-10-31", False, "미확정")] * n         # 미확정은 제외
        + [("v1", "2026-09-01", False, "판매됨")] * n * 10    # 기간 밖은 제외
    )
    write_labels(retrain, tmp_path, monkeypatch, rows)
    lift, n_labels = retrain.online_lift(date(2026, 11, 2))
    assert n_labels == 2 * n and lift == 1.0


def test_lift_needs_enough_labels(retrain, tmp_path, monkeypatch):
    write_labels(retrain, tmp_path, monkeypatch, [("v1", "2026-11-01", True, "판매됨"), ("v1", "2026-11-01", False, "미판매")])
    assert retrain.online_lift(date(2026, 11, 2)) == (None, 2)


def test_bad_lift_is_alert_not_retrain_trigger(retrain):
    assert retrain.lift_alert(-0.05) and retrain.lift_alert(0.0)
    assert not retrain.lift_alert(0.1) and not retrain.lift_alert(None)
