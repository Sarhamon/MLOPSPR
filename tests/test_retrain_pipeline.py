"""재학습 파이프라인 통합 테스트 (모델을 실제로 학습해서 20초 정도 걸림).

저장소의 cars_processed.csv 에서 매물 번호 하위 70%만 학습한 '어제의 챔피언'을 만들어 두고,
나머지 30%를 챔피언이 본 적 없는 새 매물로 삼아 04_retrain.py 의 흐름을 확인한다.
"""

import shutil

import joblib
import pytest

from conftest import ROOT


@pytest.fixture
def workdir(retrain, tmp_path, monkeypatch):
    shutil.copy(ROOT / "cars_processed.csv", tmp_path / "cars_processed.csv")
    monkeypatch.chdir(tmp_path)  # 04 는 현재 폴더 기준으로 모델/레지스트리를 읽고 쓴다

    m = retrain.m
    df = m.load_and_clean(m.CSV_PATH)
    cut = int(m.listing_no(df["링크"]).quantile(0.7))
    old = df[m.listing_no(df["링크"]) <= cut].reset_index(drop=True)
    retrain.save_model(m.make_gbr().fit(m.transform_features(old, s := m.fit_feature_state(old)), old[m.TARGET_COL]),
                       s, old, retrain.PROD_MODEL)
    return df


def test_retrain_flow(retrain, workdir):
    df = workdir
    entry = retrain.retrain(df, force=False)
    assert entry["n_new"] >= retrain.MIN_NEW_ROWS
    assert entry["triggers"][0].startswith("데이터")
    assert entry["n_holdout"] >= retrain.MIN_HOLDOUT
    assert entry["decision"] in {"promote", "keep"}

    if entry["decision"] == "promote":
        saved = joblib.load(retrain.PROD_MODEL)
        assert saved["trained_max_no"] == int(retrain.m.listing_no(df["링크"]).max())
        assert (retrain.MODELS_DIR / f"model_{entry['champion']}.joblib").exists()  # 롤백용 보관
        # 새 챔피언은 모든 매물을 봤으므로 바로 다시 돌리면 건너뛴다
        assert retrain.retrain(df, force=False)["decision"] == "skip"


def test_force_without_new_data_keeps_champion(retrain, workdir):
    df = workdir
    m = retrain.m
    retrain.save_model(m.make_gbr().fit(m.transform_features(df, s := m.fit_feature_state(df)), df[m.TARGET_COL]),
                       s, df, retrain.PROD_MODEL)
    entry = retrain.retrain(df, force=True)
    assert entry["decision"] == "keep" and "시험용 부족" in entry["reason"]


def test_refresh_pool(retrain, workdir):
    info = retrain.refresh_pool(workdir)
    assert info["rows"] > 0
    import pandas as pd
    pool = pd.read_csv(retrain.m.POOL_CSV)
    assert {"적정가격", "할인율", "저렴", "이상매물의심"} <= set(pool.columns)
