import gzip
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest
import requests
from bs4 import BeautifulSoup

from aeroscrape.config import get_app_config
from aeroscrape.data.validate import check_row_counts, quality_metrics
from aeroscrape.pipeline.ingest_tasks import failed_pages_message, remove_old_pages
from aeroscrape.scraping import scraper
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


MEHRABAD_PAGE = Path(__file__).parent / "fixtures" / "fids_mehrabad.html"


def test_domestic_only_airport_has_no_international_tables() -> None:
    page = BeautifulSoup(MEHRABAD_PAGE.read_text(encoding="utf-8"), "html.parser")

    arrivals = scrape_table(page, "input", "فرودگاه مهرآباد")
    departures = scrape_table(page, "output", "فرودگاه مهرآباد")

    assert len(arrivals) == 3 and len(departures) == 3
    assert all(f["scheduled_time"] and f["flight_number"] for f in arrivals + departures)
    assert scrape_table(page, "internal", "فرودگاه مهرآباد") == []
    assert scrape_table(page, "external", "فرودگاه مهرآباد") == []


# ---------------------------------------------------------------------------- fetching and saving


PAGE_HTML = FIDS_PAGE.read_text(encoding="utf-8")


class FakeResponse:
    def __init__(self, text: str):
        self.text, self.content = text, text.encode("utf-8")

    def raise_for_status(self) -> None:
        pass


class FakeSession:
    """Answers by airport id in the URL: 1 has a board, 2 times out, 3 is an error page."""

    def get(self, url: str, timeout):
        assert timeout == (5, 20)
        if "/2/" in url:
            raise requests.ConnectTimeout("read timed out")
        if "/3/" in url:
            return FakeResponse("<html><body>Service unavailable</body></html>")
        return FakeResponse(PAGE_HTML)


@pytest.fixture
def three_airports(monkeypatch: pytest.MonkeyPatch) -> dict:
    config = get_app_config()
    config = config | {
        "scraper": config["scraper"]
        | {
            "airports": [
                {"name": "فرودگاه مشهد", "url": "/1/mashhad"},
                {"name": "فرودگاه یزد", "url": "/2/yazd"},
                {"name": "فرودگاه رشت", "url": "/3/rasht"},
            ],
            "delay_seconds": 0,
        }
    }
    monkeypatch.setattr(scraper, "make_session", lambda cfg: FakeSession())
    return config


def test_failed_pages_are_reported_not_silently_empty(three_airports: dict, tmp_path) -> None:
    result = scraper.scrape(tmp_path, three_airports)

    statuses = {p.airport: p.status for p in result.pages}
    assert statuses == {"فرودگاه مشهد": "ok", "فرودگاه یزد": "fetch_failed", "فرودگاه رشت": "layout_changed"}
    assert len(result.flights) == 13 and set(result.flights["airport"]) == {"فرودگاه مشهد"}
    assert [p.airport for p in result.failures] == ["فرودگاه یزد", "فرودگاه رشت"]
    assert "timed out" in result.failures[0].error


def test_pages_are_saved_and_can_be_parsed_again(three_airports: dict, tmp_path) -> None:
    first = scraper.scrape(tmp_path, three_airports)

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert [p["file"] for p in manifest["pages"]] == ["1.html.gz", None, "3.html.gz"]
    assert gzip.decompress((tmp_path / "1.html.gz").read_bytes()).decode() == PAGE_HTML

    again = scraper.reparse(tmp_path)
    pd.testing.assert_frame_equal(again.flights, first.flights)
    assert [p.status for p in again.pages] == [p.status for p in first.pages]


def test_an_empty_board_is_not_a_layout_change(page: BeautifulSoup) -> None:
    for row in page.select("table tr")[1:]:
        if row.find("td"):
            row.decompose()
    assert scraper.parse_page(str(page), AIRPORT) == []


def test_requests_retry_with_backoff() -> None:
    session = scraper.make_session({"user_agent": "AeroScrape/1.0 (test)", "retries": 3})
    retry = session.get_adapter("https://fids.airport.ir/").max_retries
    assert (retry.total, retry.backoff_factor) == (3, 1)
    assert {429, 500, 502, 503, 504} <= set(retry.status_forcelist)
    assert session.headers["User-Agent"].startswith("AeroScrape/")


# ---------------------------------------------------------------------------- batch checks


def test_a_small_batch_names_what_is_missing() -> None:
    flights = pd.DataFrame({"airport": ["فرودگاه مهرآباد"] * 40 + ["فرودگاه شيراز"] * 5})
    problems = check_row_counts(flights, {"total": 200, "per_airport": {"فرودگاه مهرآباد": 50, "فرودگاه شیراز": 5}})
    assert problems == ["45 rows in total, fewer than 200", "فرودگاه مهرآباد: 40 rows, fewer than 50"]
    assert check_row_counts(flights, {"total": 10}) == []


def test_quality_metrics_count_blank_cells() -> None:
    flights = pd.DataFrame(
        {
            "airline": ["ماهان", ""],
            "status": ["پرواز کرد", None],
            "register": ["EPMOH", "EPMOI"],
            "aircraft": ["", ""],
            "actual_time": ["06:40", None],
        }
    )
    metrics = quality_metrics(flights)
    assert metrics["null_share.airline"] == 0.5 and metrics["null_share.aircraft"] == 1.0
    assert metrics["actual_time_share"] == 0.5


def test_old_saved_pages_are_removed(tmp_path) -> None:
    for day in ("2026-10-01", "2026-10-20", "2026-11-05", "notes"):
        (tmp_path / day).mkdir()
    removed = remove_old_pages(tmp_path, keep_days=30, today=datetime(2026, 11, 6))
    assert [p.name for p in removed] == ["2026-10-01"]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["2026-10-20", "2026-11-05", "notes"]


def test_failed_pages_fail_the_run_with_their_reasons() -> None:
    assert failed_pages_message([]) is None
    message = failed_pages_message([{"airport": "فرودگاه یزد", "status": "fetch_failed", "error": "timed out"}])
    assert message == "1 airport pages failed: فرودگاه یزد: fetch_failed (timed out)"
