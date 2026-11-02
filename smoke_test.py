"""
배포된 서비스가 실제로 동작하는지 확인하는 스모크 테스트 (표준 라이브러리만 사용)

    python smoke_test.py                          # http://localhost:8000
    python smoke_test.py http://<ALB 주소>         # 배포 후 확인 (deploy 워크플로도 이걸 실행)

화면, 상태, 예측, 추천, 클릭, 잘못된 입력 거부를 차례로 확인하고, 하나라도 실패하면 종료 코드 1.
"""

import json
import sys
import urllib.error
import urllib.request

BASE = (sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000").rstrip("/")
AVANTE = {"이름": "현대 더 뉴 아반떼AD 1.6 스타일", "브랜드": "현대", "연식_년": 2018, "연식_월": 3,
          "주행거리": 68658, "연료": "가솔린", "배기량": 1598, "신차가": 17280000}


def call(path, body=None):
    data = json.dumps(body, ensure_ascii=False).encode("utf-8") if body is not None else None
    req = urllib.request.Request(BASE + path, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=15) as res:
            raw = res.read().decode("utf-8")
            is_json = raw and "json" in res.headers.get("Content-Type", "")  # 204 는 본문이 비어 있음
            return res.status, (json.loads(raw) if is_json else raw)
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8")


def main():
    failed = 0

    def check(name, ok, detail=""):
        nonlocal failed
        failed += not ok
        print(f"[{'OK' if ok else 'FAIL'}] {name} {detail}")

    code, page = call("/")
    check("화면 /", code == 200 and "추천 받기" in page)

    code, health = call("/health")
    check("상태 /health", code == 200 and health.get("pool_rows", 0) > 0,
          f"model={health.get('model_version')} pool={health.get('pool_version')} rows={health.get('pool_rows')}"
          if code == 200 else str(code))

    code, old = call("/predict", AVANTE)
    code2, new = call("/predict", {**AVANTE, "연식_년": 2024})
    ok = code == code2 == 200 and 0 < old["prediction"] < new["prediction"]
    check("예측 /predict (2018년식 < 2024년식)", ok,
          f"{old['prediction']:.1f}% < {new['prediction']:.1f}%" if ok else f"{code} {old}")

    code, rec = call("/recommend", {"최대예산": 20_000_000, "개수": 5})
    ok = code == 200 and rec["count"] > 0 and all(c["가격"] <= 20_000_000 for c in rec["cars"])
    check("추천 /recommend (2000만원)", ok, f"{rec['count']}대" if ok else f"{code} {rec}")
    if ok:
        for c in rec["cars"]:
            print(f"       {c['가격'] // 10000:>5,}만원  {'저렴' if c['저렴'] else '시세'}  {c['이름'][:28]}")

    code, _ = call("/click", {"request_id": rec.get("request_id", "x") if isinstance(rec, dict) else "x",
                              "링크": "https://example.com", "rank": 1})
    check("클릭 /click", code == 204)

    code, _ = call("/recommend", {"최대예산": -1})
    check("잘못된 입력 거부", code == 422)

    print(f"\n{'모두 통과' if not failed else f'{failed}개 실패'} ({BASE})")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
