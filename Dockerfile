# 중고차 추천 서비스 이미지 (app 과 재학습 파이프라인이 같은 이미지를 씀)
#   /app  : 코드 (이미지마다 바뀜)
#   /data : 데이터/모델/추천 후보/로그. compose 에서 이름 있는 볼륨으로 마운트하면
#           처음 한 번만 아래 초기 파일이 복사되고, 이후에는 재학습 결과가 볼륨에 계속 남는다.
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    MPLCONFIGDIR=/tmp/matplotlib \
    DATA_DIR=/data

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py recommender.py run_pipeline.py smoke_test.py \
     01_crawl_and_preprocess.py 02_price_regression.py 03_collect_labels.py 04_retrain.py ./
COPY static ./static

# 초기 데이터 (볼륨이 비어 있을 때만 쓰임)
COPY cars_processed.csv price_model.joblib recommend_pool.csv /data/
COPY models/registry.jsonl /data/models/

RUN useradd --create-home --uid 1000 appuser && mkdir -p /data/logs && chown -R appuser /data
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=4)"
CMD ["python", "app.py"]
