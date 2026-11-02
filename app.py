"""
중고차 신차대비가격 예측 API

[수정] 이전 버전("from fastapi import FastAPI.py")은 입력 12개 컬럼을
feature_columns 로 reindex 만 해서, 학습 때 쓰인 나이개월/log주행/브랜드/
이름 토큰 등 나머지 피처가 전부 0으로 들어갔습니다(학습-서빙 피처 불일치).
이제 학습 스크립트의 transform_features() 를 그대로 사용해 학습 때와
똑같은 방식으로 피처를 만듭니다. 예측 모델도 학습 스크립트가 최종 모델로
고른 GradientBoosting(saved["model"])을 사용합니다.

실행: python app.py  ->  http://localhost:8000 (사용자 화면), http://localhost:8000/docs (API 문서)
"""

import hashlib
import importlib.util
import json
import logging
import os
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------
# 설정
# ---------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "price_model.joblib"
TRAIN_SCRIPT = BASE_DIR / "02_price_regression.py"

# 01 스크립트와 같은 연료 코드 (그 외 값은 '전기', '가솔린 하이브리드' 처럼 문자열 그대로)
FUEL_CODES = {"디젤": "0", "LPG": "1", "가솔린": "2"}


# ---------------------------------------------------------
# 학습 스크립트의 피처 변환 함수 + 모델 불러오기
#   (파일명이 숫자로 시작해 import 문으로는 불러올 수 없어 importlib 사용)
# ---------------------------------------------------------
_spec = importlib.util.spec_from_file_location("price_regression", TRAIN_SCRIPT)
price_regression = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(price_regression)

from recommender import POOL_PATH, load_pool, recommend  # noqa: E402


# ---------------------------------------------------------
# 모델 / 추천 후보 (재시작 없이 교체 반영)
#   04_retrain.py 가 price_model.joblib 이나 recommend_pool.csv 를 바꾸면, 다음 요청 때
#   파일 수정 시각이 달라진 것을 보고 다시 읽는다. 새 파일을 읽다 실패하면 기존 것을 계속 쓴다.
#   요청 하나는 처음 받은 스냅샷(모델+후보+버전)만 쓰므로, 도중에 교체돼도 섞이지 않는다.
# ---------------------------------------------------------
@dataclass(frozen=True)
class Artifacts:
    model: object
    feature_state: dict
    target_col: str
    pool: pd.DataFrame
    version: str  # 어떤 모델이 낸 결과인지 로그에서 구분하기 위한 버전 (모델 파일 내용 해시)
    loaded_at: str


class ArtifactStore:
    def __init__(self):
        self._lock = threading.Lock()
        self._stamp = None
        self._current = None
        self.get()  # 시작할 때 못 읽으면 바로 실패하도록

    @staticmethod
    def _file_stamp():
        return MODEL_PATH.stat().st_mtime_ns, POOL_PATH.stat().st_mtime_ns

    def get(self) -> Artifacts:
        stamp = self._file_stamp()
        if stamp != self._stamp:
            with self._lock:
                if stamp != self._stamp:
                    self._load(stamp)
        return self._current

    def _load(self, stamp):
        try:
            saved = joblib.load(MODEL_PATH)
            current = Artifacts(
                model=saved["model"], feature_state=saved["feature_state"], target_col=saved["target_col"],
                pool=load_pool(), version=hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()[:12],
                loaded_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            )
        except Exception as e:  # noqa: BLE001 - 새 파일이 깨져도 서비스는 기존 모델로 계속
            if self._current is None:
                raise
            print(f"[모델 다시 읽기 실패, 기존 모델 유지] {e!r}", file=sys.stderr)
            self._stamp = stamp  # 같은 깨진 파일을 요청마다 다시 읽지 않도록
            return
        if self._current is not None:
            print(f"[모델/추천 후보 다시 읽음] {self._current.version} -> {current.version}", file=sys.stderr)
        self._current, self._stamp = current, stamp


artifacts = ArtifactStore()


# ---------------------------------------------------------
# 요청 로그 (입력 / 예측·추천 결과 / 클릭)
#   한 요청 = JSON 한 줄. 표준 출력으로 내보내면 AWS(ECS awslogs 드라이버,
#   EC2 CloudWatch Agent 등)에서 그대로 CloudWatch Logs 로 수집되고,
#   Logs Insights 에서 필드 단위로 조회할 수 있다.
#   로컬에서는 REQUEST_LOG_FILE 에도 같은 줄을 쌓아 03_collect_labels.py 가 읽는다.
#   (컨테이너처럼 파일이 필요 없으면 REQUEST_LOG_FILE= 로 비워 두면 된다)
# ---------------------------------------------------------
REQUEST_LOG_FILE = os.environ.get("REQUEST_LOG_FILE", str(BASE_DIR / "logs" / "requests.jsonl"))

request_logger = logging.getLogger("car_service.requests")
request_logger.setLevel(logging.INFO)
request_logger.propagate = False
request_logger.addHandler(logging.StreamHandler(sys.stdout))
if REQUEST_LOG_FILE:
    Path(REQUEST_LOG_FILE).parent.mkdir(parents=True, exist_ok=True)
    request_logger.addHandler(logging.FileHandler(REQUEST_LOG_FILE, encoding="utf-8"))


def log_event(event: str, model_version: str, **fields):
    record = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "event": event,
        "model_version": model_version,
        **fields,
    }
    request_logger.info(json.dumps(record, ensure_ascii=False, default=str))


# ---------------------------------------------------------
# FastAPI 앱
# ---------------------------------------------------------
app = FastAPI(
    title="중고차 가격 예측 API",
    description="차량 이름, 연식, 주행거리, 배기량 등의 정보를 이용해 중고차 신차대비가격(%)을 예측합니다.",
    version="1.1.0",
)


# ---------------------------------------------------------
# 입력 데이터 형식
#   괄호가 들어간 컬럼명은 파이썬 변수명으로 못 쓰므로 alias 로 받습니다.
# ---------------------------------------------------------
class CarInput(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    이름: str = Field(..., examples=["현대 더 뉴 아반떼AD 1.6 스타일"])
    브랜드: str = Field(..., examples=["현대"])
    연식_년: int = Field(..., examples=[2019])
    연식_월: int = Field(..., ge=1, le=12, examples=[3])
    주행거리: float = Field(..., examples=[68658])
    연료: str = Field(..., examples=["가솔린"], description="디젤 / LPG / 가솔린 / 전기 / 가솔린 하이브리드 등")
    배기량: float = Field(..., examples=[1598])
    신차가: float | None = Field(None, examples=[17280000], description="신차 가격(옵션 미포함, 원). 모르면 비워 두세요.")

    옵션_선루프: int = 0
    옵션_파노라마선루프: int = 0
    옵션_열선앞: int = 0
    옵션_열선뒤: int = 0
    옵션_전방센서: int = 0
    옵션_후방센서: int = 0
    옵션_전방캠: int = 0
    옵션_후방캠: int = 0
    옵션_어라운드뷰: int = 0
    옵션_네비순정: int = 0

    소유자변경횟수: float = 0
    사고상세_전손: float = 0
    사고상세_침수전손: float = 0
    사고상세_침수분손: float = 0
    사고상세_도난: float = 0
    보험_내차피해_횟수: float = Field(0, alias="보험_내차피해(횟수)")
    보험_내차피해_가격: float = Field(0, alias="보험_내차피해(가격)")
    사고상세_타차가해_횟수: float = Field(0, alias="사고상세_타차가해(횟수)")
    보험_타차피해_가격: float = Field(0, alias="보험_타차피해(가격)")


def to_training_row(car: CarInput) -> pd.DataFrame:
    """API 입력을 cars_processed.csv 와 같은 형식의 한 행으로 바꾼다."""
    row = car.model_dump(by_alias=True)
    year, month = row.pop("연식_년"), row.pop("연식_월")
    row["연식"] = 12 * (2022 - year) + month  # 01 스크립트의 연식 인코딩과 동일
    row["연료"] = FUEL_CODES.get(row["연료"], row["연료"])
    return pd.DataFrame([row])


# ---------------------------------------------------------
# 사용자용 화면 + 상태 확인용 API
# ---------------------------------------------------------
@app.get("/", include_in_schema=False)
def index():
    return FileResponse(BASE_DIR / "static" / "index.html")


@app.get("/health")
def health():
    art = artifacts.get()
    return {
        "message": "중고차 가격 예측 API가 정상적으로 실행 중입니다.",
        "target": art.target_col,
        "model_version": art.version,
        "pool_rows": len(art.pool),
        "loaded_at": art.loaded_at,
    }


# ---------------------------------------------------------
# 예측 API
# ---------------------------------------------------------
@app.post("/predict")
def predict(car: CarInput):
    started = time.perf_counter()
    art = artifacts.get()
    X = price_regression.transform_features(to_training_row(car), art.feature_state)
    prediction = art.model.predict(X)[0]

    request_id = uuid.uuid4().hex
    log_event(
        "predict", art.version, request_id=request_id, input=car.model_dump(by_alias=True),
        prediction=float(prediction), latency_ms=round((time.perf_counter() - started) * 1000, 1),
    )
    return {
        "request_id": request_id,
        "prediction": float(prediction),
        "target": art.target_col,
        "message": "예측이 완료되었습니다.",
    }


# ---------------------------------------------------------
# 추천 API: 예산 안에서 시세 대비 가장 싸게 나온 매물
# ---------------------------------------------------------
class RecommendInput(BaseModel):
    최대예산: float = Field(..., gt=0, examples=[20000000], description="원 단위")
    개수: int = Field(10, ge=1, le=50)
    브랜드: str | None = Field(None, examples=["현대"])
    최소연식: int | None = Field(None, examples=[2019])
    최대주행거리: float | None = Field(None, examples=[80000])
    연료: str | None = Field(None, examples=["가솔린"])
    무사고만: bool = False


@app.post("/recommend")
def recommend_cars(req: RecommendInput):
    started = time.perf_counter()
    art = artifacts.get()
    cars = recommend(
        art.pool, req.최대예산, top_n=req.개수, brand=req.브랜드, min_year=req.최소연식,
        max_km=req.최대주행거리, fuel=req.연료, no_accident=req.무사고만,
    )

    # 추천 결과는 '나중에 정답(실제 판매 여부)을 붙일 수 있는' 최소 정보만 남김
    request_id = uuid.uuid4().hex
    log_event(
        "recommend", art.version, request_id=request_id, input=req.model_dump(),
        results=[
            {"rank": i + 1, "링크": c["링크"], "가격": c["가격"], "적정가격": c["적정가격"],
             "할인율": c["할인율"], "시세백분위": c["시세백분위"], "저렴": c["저렴"]}
            for i, c in enumerate(cars)
        ],
        latency_ms=round((time.perf_counter() - started) * 1000, 1),
    )
    return {
        "request_id": request_id,
        "count": len(cars),
        "cars": cars,
        "message": "조건에 맞는 매물이 없습니다. 예산이나 조건을 넓혀 보세요." if not cars else "추천이 완료되었습니다.",
    }


class ClickInput(BaseModel):
    request_id: str = Field(..., max_length=64)
    링크: str = Field(..., max_length=300)
    rank: int = Field(..., ge=1, le=50)


@app.post("/click", status_code=204)
def click(ev: ClickInput):
    """사용자가 추천 매물 링크를 눌렀다는 신호 (암묵적 피드백)."""
    log_event("click", artifacts.get().version, request_id=ev.request_id, 링크=ev.링크, rank=ev.rank)


# ---------------------------------------------------------
# 실행
# ---------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
