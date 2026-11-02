"""
필요할 때 실행하는 파이프라인: 크롤링(01) -> 정답 수집(03) -> 검증/재학습/추천 후보 갱신(04)

- 단계마다 별도 프로세스로 실행하고, 한 단계라도 실패하면 거기서 멈춘다
  (크롤링이 실패했는데 재학습하거나, 정답 수집이 실패했는데 성능 트리거를 판단하지 않도록).
- 04 의 데이터 검증 실패(종료 코드 2)는 프로그램 오류와 구분해 'blocked' 로 기록한다.
- 단계별 결과(상태, 소요 시간, 종료 코드)를 app.py 로그와 같은 JSON 한 줄 형식으로
  logs/pipeline.jsonl 에 남기고, 각 단계의 전체 출력은 logs/pipeline/<실행ID>_<단계>.log 에 저장한다.
- 잠금 파일(logs/pipeline.lock)로 두 번 동시에 돌지 않게 한다. 크롤링에 10분 넘게 걸려
  앞 실행이 길어지면 다음 실행과 겹칠 수 있기 때문. 오래된 잠금(LOCK_STALE_HOURS)은 무시한다.

실행
    python run_pipeline.py              # 전체
    python run_pipeline.py --skip-crawl # 크롤링 없이 03 -> 04 만 (사이트에 요청을 보내지 않고 시험할 때)
종료 코드: 모두 성공 0, 그 외 1  (나중에 스케줄러로 돌리게 되더라도 실패를 알 수 있도록)
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = BASE_DIR / "logs"
RUN_LOG = LOG_DIR / "pipeline.jsonl"
LOCK = LOG_DIR / "pipeline.lock"
LOCK_STALE_HOURS = 6

EXIT_BLOCKED = 2  # 04_retrain.py 의 데이터 검증 실패

STEPS = [
    # (단계 이름, 스크립트, 제한 시간(초))
    ("crawl", "01_crawl_and_preprocess.py", 60 * 60),
    ("labels", "03_collect_labels.py", 10 * 60),
    ("retrain", "04_retrain.py", 30 * 60),
]


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def log_run(**fields):
    LOG_DIR.mkdir(exist_ok=True)
    line = json.dumps({"ts": now(), **fields}, ensure_ascii=False)
    with RUN_LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)


def acquire_lock(run_id: str) -> bool:
    LOG_DIR.mkdir(exist_ok=True)
    if LOCK.exists() and time.time() - LOCK.stat().st_mtime > LOCK_STALE_HOURS * 3600:
        LOCK.unlink()  # 이전 실행이 비정상 종료돼 남은 잠금
    try:
        fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    with os.fdopen(fd, "w") as f:
        f.write(json.dumps({"run_id": run_id, "pid": os.getpid(), "started": now()}))
    return True


def run_step(run_id: str, name: str, script: str, timeout: int) -> str:
    out_path = LOG_DIR / "pipeline" / f"{run_id}_{name}.log"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}  # 한글/기호 출력이 콘솔 인코딩 때문에 깨지지 않도록

    started = time.perf_counter()
    with out_path.open("w", encoding="utf-8") as out:
        try:
            code = subprocess.run(
                [sys.executable, script], cwd=BASE_DIR, env=env, stdout=out, stderr=subprocess.STDOUT,
                timeout=timeout,
            ).returncode
        except subprocess.TimeoutExpired:
            code = None
    seconds = round(time.perf_counter() - started, 1)

    if code == 0:
        status = "ok"
    elif code is None:
        status = "timeout"
    elif name == "retrain" and code == EXIT_BLOCKED:
        status = "blocked"  # 데이터 검증 실패: 재학습/추천 후보 갱신이 의도적으로 멈춤
    else:
        status = "failed"
    log_run(event="pipeline_step", run_id=run_id, step=name, status=status, exit_code=code,
            seconds=seconds, output=out_path.relative_to(BASE_DIR).as_posix())
    return status


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-crawl", action="store_true", help="크롤링 없이 03 -> 04 만 실행")
    args = parser.parse_args()

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    if not acquire_lock(run_id):
        log_run(event="pipeline", run_id=run_id, status="skipped", reason=f"이미 실행 중 ({LOCK})")
        sys.exit(1)

    try:
        steps = [s for s in STEPS if not (args.skip_crawl and s[0] == "crawl")]
        log_run(event="pipeline", run_id=run_id, status="started", steps=[s[0] for s in steps])
        started = time.perf_counter()
        final = "ok"
        for name, script, timeout in steps:
            status = run_step(run_id, name, script, timeout)
            if status != "ok":
                final = status
                break
        log_run(event="pipeline", run_id=run_id, status=final, seconds=round(time.perf_counter() - started, 1))
    finally:
        LOCK.unlink(missing_ok=True)

    sys.exit(0 if final == "ok" else 1)


if __name__ == "__main__":
    main()
