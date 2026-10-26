서비스 정의
===========
중고차 구매 희망자가 최대 예산을 입력하면, 현재 판매 중인 매물 중 시세 대비 가장 저렴한
차량을 추천 이유와 함께 보여준다.

- 시세(적정가) = 모델이 예측한 신차대비가격(%) x 신차가
- 추천 순서 = 비슷한 차(국산/수입 x 연식대) 안에서 시세 대비 싸게 나온 순 (시세백분위가 낮은 순)
  - 할인율((적정가 - 판매가) / 적정가) 순으로 하면 신차대비가 낮은 오래된 차의 예측 오차가 큰 할인율로
    부풀려져, 상위 추천의 80%가 10년 넘은 차였음 (전체 매물 중 35%). 그룹 내 순위로 바꾼 뒤 34%
- '저렴' = 비슷한 차 중 하위 10% 가격 (기존: 모든 차에 같은 오차 기준을 써서 수입차에 '저렴'이 몰렸음)
- 적정가의 절반 이하로 나온 매물은 데이터 오류(미끼 매물, 신차가 오류 등) 의심으로 제외

실행 순서
=========
1) pip install -r requirements.txt
2) python 01_crawl_and_preprocess.py
   -> cars_processed.csv 생성 (이미 있으면 새 매물을 덧붙여 누적)
3) python 02_price_regression.py
   -> price_model.joblib(예측 모델), recommend_pool.csv(추천 후보 + 적정가) 생성
4) python app.py
   -> http://localhost:8000 (사용자 화면), http://localhost:8000/docs (API 문서)
   - POST /recommend : 예산/조건 -> 추천 매물 목록
   - POST /predict   : 차량 정보 -> 신차대비가격(%) 예측

테스트
======
    pip install -r requirements-dev.txt
    pytest            # 약 30초 (재학습 통합 테스트가 모델을 실제로 학습함)
- GitHub 에 push 하거나 PR 을 열면 .github/workflows/ci.yml 이 같은 테스트를 자동으로 실행합니다.
- 테스트는 logs/, models/, 모델 파일을 건드리지 않습니다 (임시 폴더 사용, 요청 로그 끔).
- 확인하는 것: 크롤러 파싱(리스 매물 구분 등), 피처 변환(학습-서빙 일치), 추천 필터/정렬,
  정답 판정, 데이터 검증, API, 재학습 파이프라인(교체/유지/건너뜀, 추천 후보 갱신)

입력/예측값 및 정답 수집
========================
- app.py 는 요청마다 JSON 한 줄을 표준 출력과 logs/requests.jsonl 에 남깁니다.
  - event: recommend(입력 조건 + 추천 매물/적정가/할인율), predict(입력 + 예측값), click(사용자가 누른 매물)
  - 공통 필드: ts(UTC), request_id, model_version(모델 파일 해시)
  - 파일 경로는 환경변수 REQUEST_LOG_FILE 로 바꿀 수 있고, 비워 두면 표준 출력에만 남깁니다.
- AWS 에 배포하면 표준 출력이 CloudWatch Logs 로 수집됩니다(ECS awslogs 드라이버 / EC2 CloudWatch Agent).
  Logs Insights 예시:
    fields ts, request_id, `input.최대예산`
    | filter event = "recommend"
    | sort ts desc
- 정답(실제 판매 여부): 크롤러를 다시 돌린 뒤 실행
    python 03_collect_labels.py [로그파일]
  -> logs/recommend_labels.csv. 추천한 매물이 다음 크롤링에서 판매완료/사라짐이면 '판매됨'.
     '저렴' 추천의 판매율이 '시세 수준' 추천보다 높은지가 1차 평가 지표입니다.
- 로그/정답 형식 예시는 samples/ 폴더를 참고하세요 (운영 로그 logs/ 는 git에 올리지 않음).

재학습 파이프라인 (챔피언 / 챌린저)
===================================
필요할 때(데이터를 새로 받고 모델/추천을 갱신하고 싶을 때) 아래 순서로 실행합니다.
run_pipeline.py 가 세 단계를 차례로 실행합니다.
    python 01_crawl_and_preprocess.py   # 새 매물 수집 (누적)
    python 03_collect_labels.py         # 로그에 정답(판매 여부) 붙이기
    python 04_retrain.py                # 데이터 검증 -> 트리거 확인 -> 재학습 -> 비교 -> 교체/유지 -> 추천 후보 갱신

0) 데이터 검증: 마지막 크롤링의 매물 수, 신차대비가격/가격이 있는 비율, 값 범위, 결측, 날짜를 점검.
   기준을 벗어나면 재학습과 추천 후보 갱신을 모두 멈추고 종료 코드 2로 끝냄 (기준은 04_retrain.py 의 CHECKS)
1) 트리거: 하나라도 해당하면 재학습
   - 데이터: 현재 모델이 학습한 뒤 새로 올라온 매물 50건 이상
   - 성능: 로그 정답 50건 이상에서 '저렴' 추천 판매율이 '시세 수준'보다 높지 않음
   - 수동: --force
2) 평가: 새 매물 중 가장 최근 절반을 시험용으로 떼어(시간순 분리), 현재 모델(챔피언)과
   새 모델(챌린저)을 같은 시험용으로 비교. 챔피언은 이 매물들을 본 적이 없어 공정한 비교가 됨
3) 교체: 챌린저 MAE가 더 낮을 때만 전체 데이터로 다시 학습해 price_model.joblib 을 교체.
   이전 모델은 models/model_<버전>.joblib 에 보관(롤백용)
4) 추천 후보 갱신: 교체 여부와 관계없이 최신 크롤링으로 recommend_pool.csv 를 다시 만듦
   (모델이 유지되는 날에도 팔린 매물은 빠지고 새 매물이 들어가도록)
5) 기록: 실행할 때마다 결과(검증 실패/건너뜀/유지/교체, 두 모델 점수, 데이터 통계)를
   models/registry.jsonl 에 한 줄씩 남김
- 실행 중인 app.py 는 모델/추천 후보 파일이 바뀌면 다음 요청 때 자동으로 다시 읽음(재시작 불필요).
  새 파일을 읽다 실패하면 기존 모델을 계속 쓰고, 현재 버전은 GET /health 에서 확인

파이프라인 한 번에 실행
=======================
    python run_pipeline.py              # 01 -> 03 -> 04
    python run_pipeline.py --skip-crawl # 크롤링 없이 03 -> 04 (사이트에 요청 없이 시험할 때)
- 한 단계라도 실패하면 거기서 멈추고 종료 코드 1. 데이터 검증 실패는 'blocked' 로 따로 표시
- 단계별 결과: logs/pipeline.jsonl (JSON 한 줄씩), 단계별 전체 출력: logs/pipeline/<실행ID>_<단계>.log
- 잠금 파일(logs/pipeline.lock)로 동시에 두 번 돌지 않음 (6시간 지난 잠금은 무시)
- 크롤링은 사이트에 약 6천 번 요청하므로 자주 돌리지 마세요.
- 추천 후보는 마지막 실행 시점의 매물 기준입니다. 실행 간격이 길면 그사이 팔린 매물이 추천될 수
  있으니, 서비스를 쓰기(시연 등) 전에 한 번 돌려 두세요.
- --skip-crawl 로 돌릴 때 마지막 크롤링이 2일보다 오래됐으면 데이터 검증(신선도)에서 막힙니다.
  오래된 매물로 추천 후보를 만들지 않기 위한 것이니, 크롤링부터 다시 하세요.

(아래는 원본 스크립트를 고칠 때 남긴 기록입니다. 02 단계는 현재 R 대신
02_price_regression.py 로 대체되었습니다.)

수정한 버그 요약
================
[01_crawl_and_preprocess.py]
1. pd.concat(...) 괄호 미종료 + 백틱(`) 오타로 인한 SyntaxError 수정
2. pd.DataFrame() 생성이 for 루프 안에 있어 매 반복마다 새로 만들어지고
   이후 append가 DataFrame 객체에 대해 호출되던 구조적 버그 수정
   -> 모든 행을 파이썬 리스트에 모은 뒤 루프 종료 후 한 번만 DataFrame 생성
3. BeautifulSoup의 find(text=...)는 최신 버전에서 find(string=...)로 변경됨
   -> string= 우선 시도, 실패 시 text= 폴백
4. requests.adapters.DEFAULT_RETRIES를 루프 안에서 매번 대입해도 실제
   재시도에 반영되지 않던 부분 -> requests.Session + HTTPAdapter(Retry)로 교체
5. User-Agent 헤더와 요청 간 딜레이(0.5초) 추가로 차단 가능성 완화
6. 각 파싱 단계에 None 체크/예외 처리 추가 (페이지 구조가 다른 매물을
   만나도 스크립트 전체가 죽지 않고 해당 건만 건너뜀)
7. changeyear/engine 등 문자열 파싱 함수에 방어 코드 추가

[02_price_regression.R]
1. cars_processed.csv를 읽어 used_car를 만드는 코드 자체가 원본에 없어서
   실행하면 바로 멈추던 문제 -> read.csv 로 로드하는 단계 추가
2. 숫자형이어야 할 컬럼이 문자로 읽힐 수 있어 as.numeric 강제 변환,
   결측 행 제거(complete.cases) 추가
3. regnewdata 수식에서 "연료_0+ +연식" 처럼 + 가 중복된 오타 수정
4. forward selection이 이미 full model(regnewdata)에서 시작해
   사실상 변수를 추가하지 못하던 문제 -> null model에서 시작하도록
   scope 지정
5. 관측치 수(559)가 하드코딩돼 있던 부분 -> nrow(newdata)로 대체
6. 영향점 행 번호(101,106,226,...)가 하드코딩돼 있어 크롤링 결과가
   달라지면 무효해지는 문제 -> H_i(레버리지) > 0.2 기준으로 동적 계산
7. 0으로 나누기 방지 로직(ifelse)에 NA 값도 함께 처리하도록 보강
8. tmp[,-c(8,36,34,32)] 처럼 인덱스 번호로 컬럼을 골라내던 부분을
   컬럼명 기준으로 변경 (실제 드롭 대상 컬럼은 데이터 확인 후 조정 필요)
9. 패키지 미설치 시 자동 설치하도록 상단에 설치 체크 추가

남은 주의사항
=============
- 보배드림은 크롤링을 막을 수 있습니다(과도한 요청 시 IP 차단, 캡차 등).
  실행 전 로봇배제표준(robots.txt)과 이용약관을 확인하세요.
- 사이트 HTML 구조가 바뀌면 find() 대상 클래스명/텍스트가 더 이상
  일치하지 않을 수 있습니다. 그럴 경우 parse_detail() 함수를 다시
  점검해야 합니다.
- 02_price_regression.R의 "8. 컬럼 드롭" 부분은 원본이 숫자 인덱스로
  컬럼을 골랐던 걸 이름 기준으로 바꾼 추정치입니다. 실제로 어떤 컬럼을
  빼려던 것인지(이름/링크/가격/연료 등) cars_processed.csv 컬럼 순서를
  보고 확인해 주세요.
