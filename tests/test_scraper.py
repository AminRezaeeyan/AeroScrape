from pathlib import Path

import pytest
from bs4 import BeautifulSoup

from aeroscrape.scraping.scraper import LayoutChangedError, scrape_table

FIDS_PAGE = Path(__file__).parent / "fixtures" / "fids_mashhad.html"
AIRPORT = "فرودگاه مشهد"


@pytest.fixture
def page() -> BeautifulSoup:
    return BeautifulSoup(FIDS_PAGE.read_text(encoding="utf-8"), "html.parser")


@pytest.mark.parametrize(
    ("table_id", "flight_type", "rows"),
    [
        ("input", "Domestic Arrival", 3),
        ("output", "Domestic Departure", 4),
        ("internal", "International Arrival", 3),
        ("external", "International Departure", 3),
    ],
)
def test_every_table_type_is_parsed(page: BeautifulSoup, table_id: str, flight_type: str, rows: int) -> None:
    flights = scrape_table(page, table_id, AIRPORT)

    assert len(flights) == rows
    assert {f["flight_type"] for f in flights} == {flight_type}
    assert all(f["airport"] == AIRPORT and f["flight_number"] for f in flights)


def test_columns_are_mapped_by_header(page: BeautifulSoup) -> None:
    departure = scrape_table(page, "output", AIRPORT)[0]

    assert departure["scheduled_time"] == "دوشنبه 06:20"
    assert departure["airline"] == "ماهان"
    assert departure["flight_number"] == "IRM1036"
    assert departure["counter"] == "5-8"
    assert departure["actual_time"] == "06:35"
    assert departure["aircraft"] == "747-400"


def test_international_departures_without_a_counter_column(page: BeautifulSoup) -> None:
    departure = scrape_table(page, "external", AIRPORT)[0]

    assert departure["flight_number"] == "MRJ4845"
    assert departure["counter"] == ""
    assert departure["actual_time"] == "09:12"


def test_blank_airline_falls_back_to_logo_code(page: BeautifulSoup) -> None:
    flights = {f["flight_number"]: f for f in scrape_table(page, "output", AIRPORT)}
    assert flights["IRM9999"]["airline"] == "IRM"


def test_missing_table_returns_nothing() -> None:
    assert scrape_table(BeautifulSoup("<div id='other'></div>", "html.parser"), "output", AIRPORT) == []


def test_unknown_header_raises(page: BeautifulSoup) -> None:
    page.find("div", id="output").find("th", class_="airlineinfo").find("a").string = "something new"
    with pytest.raises(LayoutChangedError, match="something new"):
        scrape_table(page, "output", AIRPORT)
