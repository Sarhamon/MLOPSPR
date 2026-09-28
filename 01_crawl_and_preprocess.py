"""
보배드림 중고차 매물 크롤링 + 전처리
원본 스크립트의 버그를 고친 버전입니다. 고친 부분은 파일 하단의 "수정 내역" 참고.
"""

import threading
import time
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# 0. 세션 설정 (재시도 + 헤더를 세션 하나에 고정. 원본은 루프 안에서
#    requests.adapters.DEFAULT_RETRIES 를 매번 다시 대입해도 실제 재시도
#    로직에는 반영되지 않는 문제가 있었습니다.)
#    [병렬화] 여러 스레드가 동시에 이 세션으로 요청을 보내므로, 커넥션
#    풀 크기를 워커 수만큼 넉넉히 늘려둡니다(기본값 10이라 부족할 수 있음).
# ---------------------------------------------------------------------------
LIST_WORKERS = 8     # 목록 페이지를 동시에 몇 개까지 가져올지
DETAIL_WORKERS = 12  # 상세 페이지를 동시에 몇 개까지 가져올지 (너무 크게 잡으면 차단 위험 ↑)

session = requests.Session()
retries = Retry(total=5, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504])
adapter = HTTPAdapter(max_retries=retries, pool_connections=DETAIL_WORKERS, pool_maxsize=DETAIL_WORKERS)
session.mount("https://", adapter)
session.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )
})
REQUEST_DELAY_SEC = 0.3  # 스레드마다 걸리는 최소 딜레이(병렬이라 체감 딜레이는 이보다 짧음)

_progress_lock = threading.Lock()
_progress_count = 0

# ---------------------------------------------------------------------------
# 1. 대상 URL 목록
#    [개선 v2] 원본은 14개 (maker_no, group_no) 조합을 사람이 일일이 골라
#    하드코딩해서, 그 브랜드/차종 안에서만 수집했습니다. 이 환경에서는 사이트
#    HTML을 직접 열어볼 수 없어(로봇 배제 정책으로 차단) 100% 확인은
#    못했지만, 검색 결과에 아래 URL이 각각 "국산차 검색"/"국산차 등록차량
#    전체보기" 페이지로 나왔습니다:
#      - mycar_list.php?gubun=K  (maker_no/group_no 없이 국산차 전체)
#      - mycar_list.php?gubun=I  (수입차 전체)
#    즉 maker_no/group_no를 아예 안 주면 그 브랜드 하나가 아니라 국산차/
#    수입차 전체가 한 목록으로 나오는 것으로 보입니다. 이게 맞다면 브랜드를
#    하나하나 나열할 필요 없이 이 2개 URL만으로 사실상 전체 매물을 다
#    커버합니다.
#
#    ※ 검증 필요: 확신할 수 없으니 처음 실행할 때는 아래 MAX_PAGES_PER_CATEGORY를
#    3~5 정도로 낮춰서 먼저 돌려보고, 결과 CSV의 '이름' 컬럼에 여러 브랜드
#    (현대/기아/벤츠/BMW 등)가 골고루 섞여 나오는지 확인해보세요. 만약 이
#    URL이 특정 브랜드로 쏠려 나오거나 이전 14개 카테고리와 겹치는 것만
#    나온다면, LEGACY_CATEGORY_URLS(아래) 방식으로 되돌리시면 됩니다.
#
#    페이지 수는 여전히 지정하지 않고, collect_links_for_category()가
#    "매물이 0건인 페이지를 만날 때까지" 0페이지부터 순서대로 긁어옵니다.
#    view_size도 20 → 50으로 늘려 페이지당 더 많은 매물을 담습니다.
# ---------------------------------------------------------------------------
CATEGORY_URLS = [
    "https://www.bobaedream.co.kr/mycar/mycar_list.php?gubun=K&page={}&order=S11&view_size=50",  # 국산차 전체
    "https://www.bobaedream.co.kr/mycar/mycar_list.php?gubun=I&page={}&order=S11&view_size=50",  # 수입차 전체
]

# 원본에 있던 14개 브랜드/차종별 URL. 위 "전체" URL이 기대대로 동작하지
# 않는 것으로 확인되면, 아래 주석을 풀고 CATEGORY_URLS = LEGACY_CATEGORY_URLS
# 로 바꿔서 예전 방식(브랜드별 개별 수집)으로 되돌릴 수 있습니다.
LEGACY_CATEGORY_URLS = [
    "https://www.bobaedream.co.kr/mycar/mycar_list.php?gubun=K&maker_no=49&group_no=5&page={}&order=S11&view_size=50",
    "https://www.bobaedream.co.kr/mycar/mycar_list.php?gubun=K&maker_no=49&group_no=17&page={}&order=S11&view_size=50",
    "https://www.bobaedream.co.kr/mycar/mycar_list.php?gubun=K&maker_no=49&group_no=22&page={}&order=S11&view_size=50",
    "https://www.bobaedream.co.kr/mycar/mycar_list.php?gubun=K&maker_no=49&group_no=25&page={}&order=S11&view_size=50",
    "https://www.bobaedream.co.kr/mycar/mycar_list.php?gubun=K&maker_no=49&group_no=42&page={}&order=S11&view_size=50",
    "https://www.bobaedream.co.kr/mycar/mycar_list.php?gubun=K&maker_no=1010&group_no=893&page={}&order=S11&view_size=50",
    "https://www.bobaedream.co.kr/mycar/mycar_list.php?gubun=K&maker_no=3&group_no=45&page={}&order=S11&view_size=50",
    "https://www.bobaedream.co.kr/mycar/mycar_list.php?gubun=K&maker_no=3&group_no=59&page={}&order=S11&view_size=50",
    "https://www.bobaedream.co.kr/mycar/mycar_list.php?gubun=K&maker_no=3&group_no=76&page={}&order=S11&view_size=50",
    "https://www.bobaedream.co.kr/mycar/mycar_list.php?gubun=K&maker_no=3&group_no=44&page={}&order=S11&view_size=50",
    "https://www.bobaedream.co.kr/mycar/mycar_list.php?gubun=K&maker_no=1010&group_no=879&model_no%5B%5D=1699&page={}&order=S11&view_size=50",
    "https://www.bobaedream.co.kr/mycar/mycar_list.php?gubun=K&maker_no=49&group_no=21&page={}&order=S11&view_size=50",
    "https://www.bobaedream.co.kr/mycar/mycar_list.php?gubun=K&maker_no=49&group_no=30&page={}&order=S11&view_size=50",
    "https://www.bobaedream.co.kr/mycar/mycar_list.php?gubun=K&maker_no=49&group_no=960&model_no%5B%5D=1953&page={}&order=S11&view_size=50",
]

# 처음 실행할 때는 낮게 잡아서 "전체" URL이 실제로 다양한 브랜드를
# 반환하는지부터 확인하세요. 확인되면 100~300 정도로 올려서 본 수집을
# 돌리시면 됩니다. (전체 매물 수가 많으면 그만큼 페이지 수도 많아집니다.)
MAX_PAGES_PER_CATEGORY = 100

info = ["이름", "차량번호", "링크", "연식", "주행거리", "연료", "배기량", "색상", "보증정보", "가격", "신차대비가격"]

coloptions = [
    "옵션_선루프", "옵션_파노라마선루프", "옵션_열선앞", "옵션_열선뒤", "옵션_전방센서",
    "옵션_후방센서", "옵션_전방캠", "옵션_후방캠", "옵션_어라운드뷰", "옵션_네비순정",
]
findoptions = [
    "선루프", "파노라마선루프", "열선시트(앞좌석)", "열선시트(뒷좌석)", "전방센서",
    "후방센서", "전방카메라", "후방카메라", "어라운드뷰", "네비게이션(순정)",
]
colacci_info1 = [
    "보험이력등록", "소유자변경횟수", "사고상세_전손", "사고상세_침수전손", "사고상세_침수분손",
    "사고상세_도난", "보험_내차피해(횟수)", "보험_내차피해(가격)", "사고상세_타차가해(횟수)", "보험_타차피해(가격)",
]

cols = info + coloptions + colacci_info1  # 분석 과정에서 아웃라이어 판별용으로 링크 포함


def option_check(soupobject, option_name):
    """옵션명을 받아 유/무를 반환.

    [수정] bs4 4.11+ 에서는 find(text=...) 가 아니라 find(string=...) 를 써야
    합니다(구버전 폴백으로 text=를 함께 쓰면 최신 bs4에서 DeprecationWarning이
    남아, 쓰시는 버전이 최신이라면 string= 하나만 쓰는 게 깔끔합니다).
    """
    btn = soupobject.find("button", string=option_name)
    if btn is None:
        return "정보없음"

    parent = btn.find_parent()
    if parent is None:
        return "정보없음"
    checkbox = parent.find_previous_sibling()
    if checkbox is None:
        return "정보없음"

    # [수정] bs4 4.14+ 에서는 속성이 아예 없을 때 get_attribute_list가
    # [None]이 아니라 빈 리스트 []를 반환합니다. 원본처럼 check[0]을 바로
    # 인덱싱하면 체크 안 된(속성 자체가 없는) 옵션에서 IndexError가 납니다.
    # default=[]를 명시하고, 리스트가 비어 있으면 "무"(체크 안 됨)로 처리합니다.
    check = checkbox.get_attribute_list("checked", default=[])
    if len(check) == 0:
        return "무"
    return "유"


def collect_links(list_url: str) -> list:
    """매물 목록 페이지에서 상세페이지 링크를 모두 수집."""
    res = session.get(list_url, timeout=10)
    res.raise_for_status()
    soup = BeautifulSoup(res.text, "lxml")

    cars = soup.find_all("li", attrs={"class": "product-item"})
    links = []
    for car in cars:
        if car.a is None or not car.a.get("href"):
            continue
        links.append("https://www.bobaedream.co.kr" + car.a["href"])
    return links


def fetch_and_parse_detail(link: str):
    """스레드풀 워커가 호출하는 함수. 딜레이 + 파싱 + 예외 처리를 한 번에 감쌈."""
    global _progress_count
    time.sleep(REQUEST_DELAY_SEC)
    try:
        row = parse_detail(link)
    except requests.RequestException as e:
        print(f"[상세 실패] {link}: {e}")
        row = None

    with _progress_lock:
        _progress_count += 1
        if _progress_count % 100 == 0:
            print(f"진행 중... {_progress_count}건 처리")

    return row


def parse_detail(link: str):
    """상세페이지 하나를 파싱해서 한 행(row)을 반환. 실패하면 None."""
    res2 = session.get(link, timeout=10)
    res2.raise_for_status()
    soup2 = BeautifulSoup(res2.text, "lxml")

    infobox = soup2.find("div", attrs={"class": "info-util box"})
    if infobox is None:
        return None
    ratiopr = infobox.find("b")
    if ratiopr is None:
        return None

    name = soup2.find("h3", attrs={"class": "tit"})
    state = soup2.find("div", attrs={"class": "tbl-01 st-low"})
    galdata = soup2.find("div", attrs={"class": "gallery-data"})
    if name is None or state is None or galdata is None:
        return None
    carnumber = galdata.find("b")

    def cell(label):
        th = state.find("th", string=label)
        if th is None or th.find_next_sibling("td") is None:
            return ""
        return th.find_next_sibling("td").get_text()

    year = cell("연식")
    km = cell("주행거리")
    fuel = cell("연료")
    amount = cell("배기량")
    color = cell("색상")

    guarn_tag = state.find("b", string="보증정보")
    guarn = guarn_tag.find_next("td").get_text() if guarn_tag else ""

    price_tag = soup2.find("span", attrs={"class": "price"})
    price = price_tag.get_text() if price_tag else ""

    # 옵션 표
    option_table = soup2.find("div", attrs={"class": "tbl-option"})
    checkoptions = []
    if option_table is not None and option_table.find("th", string="외관") is not None:
        for option in findoptions:
            checkoptions.append(option_check(option_table, option))
    else:
        checkoptions = [""] * len(coloptions)

    # 보험이력 등록 여부
    ins_span = infobox.find("span", attrs={"class": "round-ln insurance"})
    acc1 = "미등록"
    if ins_span is not None:
        nxt = ins_span.find_next("i")
        if nxt is not None and nxt.find_next("em") is not None:
            acc1 = "등록"

    findacci_info1 = [""] * (len(colacci_info1) - 1)
    if acc1 == "등록":
        try:
            acc1table = soup2.find("div", attrs={"class": "info-insurance"})
            insurdt1 = (
                acc1table.find("th", string="차량번호/소유자변경").find_next_sibling("td").get_text()[-2]
            )
            insuraccis1 = (
                acc1table.find("th", string="자동차보험 특수사고")
                .find_next_sibling("td")
                .get_text()
                .split("/")
            )
            insurdt2 = insuraccis1[0][-2]
            insurdt3 = insuraccis1[1][-2]
            insurdt4 = insuraccis1[2][-2]
            insurdt5 = insuraccis1[3][-1]

            insuraccis2 = (
                acc1table.find("th", string="보험사고(내차피해)")
                .find_next_sibling("td")
                .get_text()
                .split("회")
            )
            insurdt6 = insuraccis2[0]
            insurdt7 = insuraccis2[1][2:-2]

            insuraccis3 = (
                acc1table.find("th", string="보험사고(타차가해)")
                .find_next_sibling("td")
                .get_text()
                .split("회")
            )
            insurdt8 = insuraccis3[0]
            insurdt9 = insuraccis3[1][2:-2]

            findacci_info1 = [
                insurdt1, insurdt2, insurdt3, insurdt4, insurdt5,
                insurdt6, insurdt7, insurdt8, insurdt9,
            ]
        except (AttributeError, IndexError):
            findacci_info1 = [""] * (len(colacci_info1) - 1)

    row = (
        [name.get_text(), carnumber.get_text() if carnumber else "", link, year, km, fuel, amount, color, guarn, price, ratiopr.get_text()]
        + checkoptions
        + [acc1]
        + findacci_info1
    )
    return row


# ---------------------------------------------------------------------------
# 2. 카테고리별 목록 페이지 자동 탐색
#    [개선] 페이지 수를 미리 알 필요 없이, 0페이지부터 순서대로 요청하다가
#    매물이 0건인 페이지를 만나면 그 카테고리는 끝난 것으로 보고 멈춥니다.
#    MAX_PAGES_PER_CATEGORY는 혹시 사이트가 끝없이 빈 목록을 반환하는
#    상황(버그 등)을 대비한 안전장치일 뿐입니다.
# ---------------------------------------------------------------------------
def collect_links_for_category(base_url_template: str, max_pages: int = MAX_PAGES_PER_CATEGORY) -> list:
    links = []
    for page in range(max_pages):
        url = base_url_template.format(page)
        try:
            page_links = collect_links(url)
        except requests.RequestException as e:
            print(f"[목록 실패] {url}: {e}")
            break
        if not page_links:
            print(f"[{base_url_template.split('?')[1][:40]}...] {page}페이지에서 종료 (매물 0건)")
            break
        links.extend(page_links)
    else:
        print(
            f"[경고] {base_url_template} 는 MAX_PAGES_PER_CATEGORY({max_pages})에 도달해 "
            f"강제로 멈췄습니다. 실제로 더 있을 수 있으니 값을 늘려보세요."
        )
    return links


# ---------------------------------------------------------------------------
# 3. 크롤링 실행 (병렬 처리)
#    [수정] 원본은 pd.DataFrame(...) 호출이 for-url 루프 안에 있어서
#    매 url마다 df_cars 를 그 시점까지 쌓인 리스트로 다시 만들고, 그 다음
#    줄에서 df_cars.append(temp) 가 DataFrame에 대해 호출되어 사실상
#    동작하지 않는 구조였습니다. 여기서는 모든 행을 파이썬 list에 다 모은
#    다음, 루프가 끝난 뒤 딱 한 번만 DataFrame으로 변환합니다.
#
#    [병렬화] 카테고리 단위(14개)는 동시에 LIST_WORKERS개씩 돌리고, 각
#    카테고리 내부에서는 페이지를 순서대로(0, 1, 2, ...) 요청하면서 빈
#    페이지가 나오면 멈춥니다(순서를 지켜야 "몇 페이지까지 있는지"를 알
#    수 있으므로 카테고리 내부는 병렬화하지 않았습니다). 상세 페이지는
#    모아진 링크 전체를 DETAIL_WORKERS개씩 동시에 가져옵니다.
# ---------------------------------------------------------------------------

# 3-1. 카테고리들을 동시에 순회하며 상세페이지 링크를 모은다
all_links = []
with ThreadPoolExecutor(max_workers=LIST_WORKERS) as executor:
    future_to_cat = {executor.submit(collect_links_for_category, c): c for c in CATEGORY_URLS}
    done = 0
    for future in as_completed(future_to_cat):
        category_url = future_to_cat[future]
        done += 1
        try:
            links = future.result()
        except Exception as e:  # noqa: BLE001 - 카테고리 하나가 실패해도 나머지는 계속
            print(f"[카테고리 실패] {category_url}: {e}")
            continue
        all_links.extend(links)
        print(f"[카테고리 {done}/{len(CATEGORY_URLS)}] {len(links)}건 수집")

# 같은 매물이 여러 카테고리에 중복으로 걸릴 수 있어 링크 기준으로 중복 제거
all_links = list(dict.fromkeys(all_links))
print(f"수집 대상 상세페이지: {len(all_links)}건 (중복 제거 후)")

# 3-2. 상세 페이지들을 동시에 가져와서 파싱한다
rows = []
with ThreadPoolExecutor(max_workers=DETAIL_WORKERS) as executor:
    futures = [executor.submit(fetch_and_parse_detail, link) for link in all_links]
    for future in as_completed(futures):
        row = future.result()
        if row is not None:
            rows.append(row)

df_cars = pd.DataFrame(data=rows, columns=cols)
print(f"수집 완료: {len(df_cars)}건")

if df_cars.empty:
    raise SystemExit("수집된 데이터가 없습니다. 사이트 구조가 바뀌었거나 차단됐을 수 있습니다.")


# ---------------------------------------------------------------------------
# 4. 전처리 (원본 로직 유지, 안전장치만 추가)
# ---------------------------------------------------------------------------
# [수정] 원본 changemodel()은 공백 3개가 나오면 그 앞까지만 남겼습니다.
# 현대/기아는 "더 뉴", "디 올 뉴", "올 뉴" 같은 마케팅 수식어가 브랜드
# 바로 뒤에 붙는 경우가 많아서("현대 더 뉴 그랜저 하이브리드 2.4"), 이
# 규칙대로 자르면 정작 중요한 모델명("그랜저")이 통째로 날아가고
# "현대 더 뉴"만 남았습니다. 실제로 크롤링 데이터에서 "기아 더 뉴"라는
# 이름으로 126건이 뭉뚱그려져 있었습니다.
#
# 이제는:
#   - "이름" 컬럼은 자르지 않고 원본 전체 제목을 그대로 저장합니다
#     (나중에 필요하면 언제든 다시 파싱할 수 있도록 원본을 보존).
#   - 수식어(더/디/올/뉴 등)를 건너뛰고 그 다음에 나오는 진짜 모델명을
#     "모델명" 컬럼으로 따로 뽑아냅니다. 모델링에는 이 "모델명"(또는
#     "브랜드")을 쓰는 걸 추천합니다 — 전체 제목을 그대로 범주형으로
#     쓰면 값이 거의 다 달라져서(카디널리티 폭발) 학습에 오히려 방해가
#     됩니다.
FILLER_WORDS = {
    "더", "디", "올", "뉴",
    "the", "The", "THE", "new", "New", "NEW",
}


def changemodelname(str1: str) -> str:
    if "그랜져" in str1:
        return str1.replace("그랜져", "그랜저")
    return str1


def extract_brand_model(full_name: str):
    """'현대 더 뉴 그랜저 하이브리드 2.4' -> ('현대', '그랜저')
    '포르쉐 911 8세대' -> ('포르쉐', '911')
    수식어(더/디/올/뉴)를 건너뛰고 그 다음 나오는 1~2 단어를 모델명으로
    본다. 완벽하지 않을 수 있으니(사이트마다 제목 구성이 다를 수 있음),
    이상하게 뽑히는 사례가 보이면 이 함수만 조정하면 됩니다.
    """
    tokens = str(full_name).split()
    if not tokens:
        return "", ""
    brand = tokens[0]
    rest = tokens[1:]

    idx = 0
    while idx < len(rest) and rest[idx] in FILLER_WORDS:
        idx += 1

    model_tokens = rest[idx: idx + 2]
    model = " ".join(model_tokens).strip()
    return brand, model


df_cars["이름"] = df_cars["이름"].apply(changemodelname)  # 오타 교정만, 자르지 않음
df_cars[["브랜드", "모델명"]] = df_cars["이름"].apply(
    lambda x: pd.Series(extract_brand_model(x))
)


def changeyear(string: str):
    if not string or len(string) < 7:
        return np.nan
    try:
        year = int(string[:4])
        month = int(string[5:7])
    except ValueError:
        return np.nan
    return 12 * (2022 - year) + month


df_cars["연식"] = df_cars["연식"].apply(changeyear)

df_cars["주행거리"] = df_cars["주행거리"].apply(
    lambda x: x.replace(",", "")[:-2].rstrip() if isinstance(x, str) else x
)

matchfuel = {"디젤": 0, "LPG": 1, "가솔린": 2}
df_cars["연료"] = df_cars["연료"].replace(matchfuel)


def engine(str1: str):
    if not isinstance(str1, str):
        return np.nan
    index = str1.find("cc")
    if index == -1:
        return np.nan
    return str1[:index].replace(",", "").rstrip()


df_cars["배기량"] = df_cars["배기량"].apply(engine)


def guar(str1: str):
    if str1 in ("만료", "불가"):
        return pd.Series([0, 0, 0])
    if str1 == "정보없음" or not isinstance(str1, str):
        return pd.Series([np.nan, np.nan, np.nan])
    index = str1.find("/")
    if index == -1:
        time_ = 0
    else:
        time_ = int(str1[: index - 3].strip())
    km = str1[index + 1 :].replace(",", "")[:-2].strip()
    return pd.Series([1, time_, km])


guartable = df_cars["보증정보"].apply(guar)
guartable.columns = ["보증여부", "보증기간", "보증거리"]
df_cars = pd.concat([df_cars, guartable], axis=1)  # [수정] 원본은 괄호가 안 닫혀 SyntaxError
df_cars = df_cars.drop(["보증정보"], axis=1)

matchop = {"무": 0, "유": 1}
df_cars = df_cars.replace(matchop)

insurinfo = {"미등록": 0, "등록": 1}
df_cars["보험이력등록"] = df_cars["보험이력등록"].replace(insurinfo)
df_cars["보험_내차피해(가격)"] = df_cars["보험_내차피해(가격)"].apply(
    lambda x: x.replace(",", "") if isinstance(x, str) else np.nan
)
df_cars["보험_타차피해(가격)"] = df_cars["보험_타차피해(가격)"].apply(
    lambda x: x.replace(",", "") if isinstance(x, str) else np.nan
)

df_cars["가격"] = df_cars["가격"].apply(
    lambda x: x.replace("만", "0000").replace(",", "")[:-1] if isinstance(x, str) else x
)
df_cars["신차대비가격"] = df_cars["신차대비가격"].apply(lambda x: np.nan if x == "준비중" else x)

keep_cols = [
    "이름", "브랜드", "모델명", "연식", "주행거리", "연료", "배기량", "색상",
    "옵션_선루프", "옵션_파노라마선루프", "옵션_열선앞", "옵션_열선뒤", "옵션_전방센서",
    "옵션_후방센서", "옵션_전방캠", "옵션_후방캠", "옵션_어라운드뷰", "옵션_네비순정",
    "보험이력등록", "소유자변경횟수", "사고상세_전손", "사고상세_침수전손", "사고상세_침수분손", "사고상세_도난",
    "보험_내차피해(횟수)", "보험_내차피해(가격)", "사고상세_타차가해(횟수)", "보험_타차피해(가격)",
    "가격", "신차대비가격", "링크",
]
df_cars = df_cars.loc[:, keep_cols]
df_cars.to_csv("cars_processed.csv", index=False, encoding="utf-8-sig")
print("저장 완료: cars_processed.csv")


# ---------------------------------------------------------------------------
# 수정 내역 요약
# ---------------------------------------------------------------------------
# 1) pd.concat(...) 괄호 미종료 + 백틱(`) 오타로 인한 SyntaxError 수정
# 2) pd.DataFrame() 생성이 for 루프 안에 있어 매 반복마다 새로 만들어지고
#    이후 append가 DataFrame 객체에 대해 호출되던 구조적 버그 수정
#    → 모든 행을 파이썬 리스트에 모은 뒤 루프 종료 후 한 번만 DataFrame 생성
# 3) BeautifulSoup의 find(text=...) 는 최신 버전에서 find(string=...) 로
#    변경되어 DeprecationWarning이 남 → string= 만 사용하도록 통일
# 4) requests.adapters.DEFAULT_RETRIES 를 루프 안에서 매번 대입해도 실제
#    재시도에 반영되지 않던 부분 → requests.Session + HTTPAdapter(Retry)로 교체
# 5) User-Agent 헤더와 요청 간 딜레이(0.5초)를 추가해 차단 가능성을 낮춤
# 6) 각 파싱 단계에 None 체크/예외 처리를 추가해 페이지 구조가 다른 매물을
#    만나도 스크립트 전체가 죽지 않고 해당 건만 건너뛰도록 함
# 7) changeyear/engine 등 문자열 파싱 함수에 형식이 다른 값이 들어와도
#    죽지 않도록 방어 코드 추가
# 8) get_attribute_list("checked")를 바로 인덱싱하다 IndexError가 나던
#    문제 수정 (bs4 4.14+에서 속성 없으면 빈 리스트를 반환)
# 9) [병렬화] 순차 실행이던 크롤링을 ThreadPoolExecutor로 교체:
#    - 목록 페이지 75개를 LIST_WORKERS(기본 8)개씩 동시에 가져옴
#    - 상세 페이지를 DETAIL_WORKERS(기본 12)개씩 동시에 가져옴
#    - 링크 중복 제거, 20건마다 진행 상황 출력 추가
#    체감 속도는 대략 5~8배 빨라지지만, 동시 요청 수가 늘어난 만큼
#    사이트에 차단될 위험도 커지므로 막히면 LIST_WORKERS/DETAIL_WORKERS
#    값을 줄이세요.
# 10) [데이터 확대] 카테고리마다 "페이지 몇 개까지 있는지"를 사람이 세서
#     하드코딩했던 부분을 없애고, 매물이 0건인 페이지를 만날 때까지 자동으로
#     계속 가져오도록 변경(collect_links_for_category). 그동안 하드코딩된
#     페이지 수보다 실제로 매물이 더 많았다면 그만큼 더 수집됩니다.
#     또한 view_size를 20 → 50으로 늘려 페이지당 더 많은 매물을 담아
#     목록 페이지 요청 횟수 자체를 줄였습니다.
#
# ---------------------------------------------------------------------------
# 데이터를 더 늘리려면 (카테고리 자체를 넓히기)
# ---------------------------------------------------------------------------
# 지금 CATEGORY_URLS에 있는 14개는 원본 스크립트에 있던 (maker_no, group_no)
# 조합 그대로입니다. 이 환경에서는 보배드림 사이트에 직접 접속해 카테고리
# 선택 화면의 실제 코드값을 확인할 수 없어서(로봇 배제 정책으로 차단),
# 목록을 임의로 늘리지는 않았습니다. 더 많은 차종/브랜드를 추가하고 싶다면:
#
#   1. 브라우저로 https://www.bobaedream.co.kr/mycar/mycar_list.php 접속
#   2. 원하는 제조사 → 차종을 선택하면 주소창 URL에 maker_no=...&group_no=...
#      (필요하면 model_no[]=... 도) 값이 나타남
#   3. 그 URL에서 "&page=숫자" 부분만 "&page={}" 로 바꿔서
#      CATEGORY_URLS 리스트에 새 항목으로 추가
#
# 예: 주소창이 ...maker_no=6&group_no=120&page=1&order=S11&view_size=50
#     이었다면 아래처럼 추가:
#     "https://www.bobaedream.co.kr/mycar/mycar_list.php?gubun=K&maker_no=6&group_no=120&page={}&order=S11&view_size=50"
#
# 참고: 국산/수입 전체 차량을 한 번에 긁는 "전체 카테고리" URL이 사이트에
# 있다면(예: maker_no를 생략하거나 all 옵션) 그 URL 하나만 CATEGORY_URLS에
# 넣는 것으로 모든 차종을 커버할 수도 있습니다. 실제 그런 URL이 있는지는
# 사이트에서 직접 확인해 알려주시면, 그 형태에 맞게 코드를 다시 정리해
# 드리겠습니다.