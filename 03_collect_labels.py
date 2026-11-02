"""
추천 로그에 정답(실제 판매 여부)을 붙이는 스크립트

app.py 가 남긴 요청 로그(JSON 한 줄씩)와, 그 이후에 다시 돌린 크롤링 결과
(cars_processed.csv)를 맞춰 보고 "추천한 매물이 실제로 팔렸는가"를 정답으로 붙인다.

판매 여부 판단 (추천 시점 이후에 크롤링이 한 번 이상 있었던 경우만)
  - 마지막 크롤링에서 가격상태가 '판매완료'      -> 판매됨
  - 마지막 크롤링 목록에서 사라짐(수집일이 더 옛날) -> 판매됨(또는 삭제)
  - 마지막 크롤링에도 판매중으로 남아 있음        -> 미판매
  - 추천 이후 크롤링이 아직 없음                  -> 미확정 (평가에서 제외)

시세보다 싸다고 추천한 차가 정말 싸다면, 다른 매물보다 빨리 팔려야 한다.
그래서 '저렴' 추천의 판매율이 '시세 수준' 추천보다 높은지가 1차 평가 지표다.

실행
    python 03_collect_labels.py                       # logs/requests.jsonl 사용
    python 03_collect_labels.py cloudwatch_export.jsonl  # CloudWatch 에서 내려받은 로그
"""

import json
import sys
from pathlib import Path

import pandas as pd

LOG_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("logs/requests.jsonl")
CRAWL_CSV = "cars_processed.csv"
OUT_CSV = "logs/recommend_labels.csv"


def read_events(path: Path) -> list[dict]:
    """JSON 한 줄씩 읽기. CloudWatch 내보내기처럼 {"message": "<json>"} 로 감싸진 줄도 처리."""
    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "event" not in ev and isinstance(ev.get("message"), str):
            try:
                ev = json.loads(ev["message"])
            except json.JSONDecodeError:
                continue
        events.append(ev)
    return events


def attach_labels(rec: pd.DataFrame, crawl: pd.DataFrame) -> pd.DataFrame:
    """추천 행(링크, 추천일)에 크롤링 결과(링크, 가격상태, 수집일)를 맞춰 판매여부를 붙인다."""
    latest = crawl["수집일"].max()
    rec = rec.merge(crawl.rename(columns={"수집일": "마지막확인일"}), on="링크", how="left")

    def label(r):
        if pd.isna(r["마지막확인일"]) or latest <= r["추천일"]:
            return "미확정"
        if r["가격상태"] == "판매완료" or r["마지막확인일"] < latest:
            return "판매됨"
        return "미판매"

    rec["판매여부"] = rec.apply(label, axis=1) if len(rec) else pd.Series(dtype=str)
    return rec


def main():
    if not LOG_PATH.exists():
        # 서비스 초기에는 정상적인 상황이라 오류로 끝내지 않음 (run_pipeline.py 가 실패로 보지 않도록)
        print(f"로그 파일이 없습니다: {LOG_PATH} (app.py 가 요청을 받은 뒤부터 정답을 붙일 수 있습니다)")
        return
    events = read_events(LOG_PATH)

    rows = []
    for ev in events:
        if ev.get("event") != "recommend":
            continue
        for r in ev.get("results", []):
            rows.append({
                # 로그 시각은 UTC, 크롤러의 수집일은 한국 날짜라 한국 시간으로 맞춰 비교
                "request_id": ev["request_id"], "model_version": ev["model_version"],
                "추천일": pd.Timestamp(ev["ts"]).tz_convert("Asia/Seoul").date().isoformat(),
                "최대예산": ev["input"].get("최대예산"), **r,
            })
    if not rows:
        print("추천 로그가 아직 없습니다.")
        return
    rec = pd.DataFrame(rows)

    clicks = {(ev["request_id"], ev["링크"]) for ev in events if ev.get("event") == "click"}
    rec["클릭"] = [int((rid, link) in clicks) for rid, link in zip(rec["request_id"], rec["링크"])]

    crawl = pd.read_csv(CRAWL_CSV, usecols=["링크", "가격상태", "수집일"])
    latest = crawl["수집일"].max()
    rec = attach_labels(rec, crawl)
    Path(OUT_CSV).parent.mkdir(parents=True, exist_ok=True)
    rec.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    print(f"추천 요청 {rec['request_id'].nunique()}건, 추천 매물 {len(rec)}건 -> {OUT_CSV}")
    print(f"마지막 크롤링: {latest}")
    print(f"\n순위별 클릭률:\n{rec.groupby('rank')['클릭'].mean().round(3).to_string()}")

    done = rec[rec["판매여부"] != "미확정"]
    if done.empty:
        print("\n추천 이후 크롤링이 아직 없어 판매 여부를 판단할 수 없습니다. "
              "01_crawl_and_preprocess.py 를 다시 돌린 뒤 실행하세요.")
        return
    sold = done["판매여부"].eq("판매됨")
    print(f"\n판매 여부 확정 {len(done)}건 중 판매됨 비율: {sold.mean():.3f}")
    print("추천 구분별 판매율 ('저렴'이 '시세 수준'보다 높아야 모델이 쓸모 있음):")
    print(sold.groupby(done["저렴"].map({True: "저렴", False: "시세 수준"})).agg(["mean", "count"]).round(3).to_string())


if __name__ == "__main__":
    main()
