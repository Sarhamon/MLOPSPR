import json

import pandas as pd

LINK = "https://www.bobaedream.co.kr/mycar/mycar_view.php?no={}&gubun=K"


def test_attach_labels(labels):
    rec = pd.DataFrame({
        "링크": [LINK.format(n) for n in (1, 2, 3, 4)],
        "추천일": ["2026-10-26"] * 4,
    })
    crawl = pd.DataFrame({
        "링크": [LINK.format(n) for n in (1, 2, 3)],
        "가격상태": ["판매완료", "판매중", "판매중"],
        "수집일": ["2026-10-27", "2026-10-26", "2026-10-27"],
    })
    out = labels.attach_labels(rec, crawl).set_index("링크")["판매여부"]
    assert out[LINK.format(1)] == "판매됨"  # 다음 크롤링에서 판매완료
    assert out[LINK.format(2)] == "판매됨"  # 다음 크롤링 목록에서 사라짐
    assert out[LINK.format(3)] == "미판매"  # 계속 판매 중
    assert out[LINK.format(4)] == "미확정"  # 크롤링 데이터에 없는 매물


def test_no_crawl_after_recommendation_is_undecided(labels):
    rec = pd.DataFrame({"링크": [LINK.format(1)], "추천일": ["2026-10-27"]})
    crawl = pd.DataFrame({"링크": [LINK.format(1)], "가격상태": ["판매완료"], "수집일": ["2026-10-27"]})
    assert labels.attach_labels(rec, crawl)["판매여부"].tolist() == ["미확정"]


def test_read_events_plain_and_cloudwatch(labels, tmp_path):
    ev = {"event": "click", "request_id": "abc", "링크": LINK.format(1), "rank": 1}
    path = tmp_path / "log.jsonl"
    path.write_text("\n".join([
        json.dumps(ev, ensure_ascii=False),
        json.dumps({"timestamp": 1, "message": json.dumps(ev, ensure_ascii=False)}),  # CloudWatch 내보내기 형식
        "INFO:     127.0.0.1 - GET /health",  # uvicorn 같은 다른 출력
        "{깨진 json",
    ]), encoding="utf-8")
    events = labels.read_events(path)
    assert len(events) == 2 and all(e["request_id"] == "abc" for e in events)
