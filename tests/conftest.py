import importlib.util
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["REQUEST_LOG_FILE"] = ""  # 테스트 중에는 logs/requests.jsonl 에 요청 로그를 남기지 않음


def load_script(filename: str):
    """숫자로 시작하는 파일명(01_..., 02_...)은 import 문으로 못 불러와서 경로로 불러온다."""
    spec = importlib.util.spec_from_file_location(Path(filename).stem.lstrip("0123456789_"), ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def crawler():
    return load_script("01_crawl_and_preprocess.py")


@pytest.fixture(scope="session")
def pr():
    return load_script("02_price_regression.py")


@pytest.fixture(scope="session")
def labels():
    return load_script("03_collect_labels.py")


@pytest.fixture(scope="session")
def retrain():
    return load_script("04_retrain.py")
