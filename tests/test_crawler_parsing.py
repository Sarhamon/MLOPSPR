"""상세 페이지 파싱. 실제 사이트 HTML 구조(2026-10 확인)를 줄여 만든 조각으로 확인한다."""

from bs4 import BeautifulSoup


def infobox(inner: str):
    return BeautifulSoup(f'<div class="info-util box"><ul>{inner}</ul></div>', "lxml").find(
        "div", attrs={"class": "info-util box"})


def test_ratio_from_pie_chart(crawler):
    box = infobox('<li><i class="tit">보험이력<em>3건</em></i></li>'
                  '<li><div class="pie-chart-detail" data-percent="65"><b class="percent">65</b></div>'
                  '<i class="tit">신차대비</i></li>')
    assert crawler.parse_ratio_box(box) == ("65", "일반")


def test_ratio_not_ready(crawler):
    box = infobox('<li><i class="tit">신차대비</i><b class="in-txt gray">준비중</b></li>')
    assert crawler.parse_ratio_box(box) == ("", "일반")


def test_lease_listing_is_not_ratio(crawler):
    """원래 버그: 첫 <b> 를 그대로 써서 '운용', '소유' 가 신차대비가격 컬럼에 들어갔다."""
    box = infobox('<li><i class="tit">보험이력<em>1건</em></i></li>'
                  '<li><a><i class="tit">리스종류</i><b class="in-txt">운용</b></a></li>')
    assert crawler.parse_ratio_box(box) == ("", "승계(리스종류 운용)")

    box = infobox('<li><a><i class="tit">만기 후</i><b class="in-txt">소유</b></a></li>')
    assert crawler.parse_ratio_box(box) == ("", "승계(만기 후 소유)")


def test_option_check(crawler):
    html = ('<div class="tbl-option">'
            '<div><input type="checkbox" checked><label><button>선루프</button></label></div>'
            '<div><input type="checkbox"><label><button>후방카메라</button></label></div></div>')
    table = BeautifulSoup(html, "lxml")
    assert crawler.option_check(table, "선루프") == "유"
    assert crawler.option_check(table, "후방카메라") == "무"
    assert crawler.option_check(table, "어라운드뷰") == "정보없음"
