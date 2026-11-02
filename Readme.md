실행 순서
=========
1) pip install requests beautifulsoup4 lxml pandas numpy
   python3 01_crawl_and_preprocess.py
   -> 같은 폴더에 cars_processed.csv 생성

2) R에서 (working directory를 cars_processed.csv가 있는 폴더로 맞춘 뒤)
   source("02_price_regression.R")

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
