"""Fetch the FIDS flight boards, keep the raw pages, and parse them into flight rows.

Fetching and parsing are separate: every page is saved (gzipped) under
data/raw_html/<date>/<time>/ with a manifest, so a parser fix can be applied to past
scrapes with `reparse` instead of scraping again, and saved pages make realistic test
fixtures. A page that can't be fetched after retries, or whose tables no longer look as
expected, is recorded as a failure instead of quietly becoming an empty list.
"""

import gzip
import json
import logging
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from aeroscrape.config import get_app_config

logger = logging.getLogger(__name__)


# Header text on fids.airport.ir mapped to our column names. Departure and arrival tables use
# different words for the same column, and international departures have no counter column.
HEADER_FIELDS = {
    "زمان برنامه ای": "scheduled_time",
    "ایرلاین": "airline",
    "شماره پرواز": "flight_number",
    "مقصد": "destination_or_origin",
    "مبدا": "destination_or_origin",
    "وضعیت": "status",
    "کانتر": "counter",
    "نقاله": "counter",
    "زمان واقعی": "actual_time",
    "تاریخ و زمان واقعی": "actual_time",
    "رجیستر": "register",
    "هواپیما": "aircraft",
    "تاریخ": "date",
}
# The "day" header has no cell of its own: day and scheduled time share the first cell
DAY_HEADER = "روز"
OUTPUT_FIELDS = (
    "scheduled_time",
    "airline",
    "flight_number",
    "destination_or_origin",
    "status",
    "counter",
    "actual_time",
    "register",
    "aircraft",
)


class LayoutChangedError(Exception):
    """The page no longer has the table structure the parser expects."""


def _header_text(th) -> str:
    label = th.find("a") or th
    return " ".join(label.get_text(" ", strip=True).split())


def _column_fields(table) -> list[str]:
    header_row = table.find("tr")
    headers = [_header_text(th) for th in header_row.find_all("th")] if header_row else []
    if headers[:1] == [DAY_HEADER]:
        headers = headers[1:]
    unknown = [h for h in headers if h not in HEADER_FIELDS]
    if not headers or unknown:
        raise LayoutChangedError(f"unexpected table headers {unknown or headers}")
    return [HEADER_FIELDS[h] for h in headers]


def _airline_text(cell) -> str:
    """Airline name, falling back to the logo's airline code when the cell has no text."""
    text = cell.get_text(" ", strip=True)
    if text:
        return text
    img = cell.find("img")
    src = img.get("src", "") if img else ""
    return src.rsplit("/", 1)[-1].split(".", 1)[0].upper()


def scrape_table(soup, table_id, airport_name):
    app_config = get_app_config()
    flight_table_ids = app_config["scraper"]["flight_table_ids"]

    table_container = soup.find("div", id=table_id)
    if not table_container:
        return []

    table = table_container.find("table")
    if not table:
        return []

    fields = _column_fields(table)
    data = []
    for row in table.find_all("tr")[1:]:
        cells = row.find_all("td")
        if not cells:
            continue
        if len(cells) != len(fields):
            raise LayoutChangedError(f"row has {len(cells)} cells but the header has {len(fields)} columns")

        values = {}
        for field_name, cell in zip(fields, cells, strict=True):
            values[field_name] = _airline_text(cell) if field_name == "airline" else cell.get_text(" ", strip=True)

        flight_data = {name: values.get(name, "") for name in OUTPUT_FIELDS}
        flight_data["airport"] = airport_name
        flight_data["flight_type"] = flight_table_ids.get(table_id, "Unknown")
        data.append(flight_data)
    return data


class PageStatus:
    OK = "ok"
    FETCH_FAILED = "fetch_failed"
    LAYOUT_CHANGED = "layout_changed"


@dataclass
class PageResult:
    airport: str
    url: str
    status: str
    rows: int = 0
    file: str | None = None  # the saved page, relative to the run directory
    error: str | None = None
    fetched_at: str | None = None


@dataclass
class ScrapeResult:
    flights: pd.DataFrame
    pages: list[PageResult] = field(default_factory=list)

    @property
    def failures(self) -> list[PageResult]:
        return [p for p in self.pages if p.status != PageStatus.OK]


def make_session(scraper_cfg: dict[str, Any]) -> requests.Session:
    """A session that retries connection errors and 429/5xx responses with exponential backoff."""
    retry = Retry(
        total=scraper_cfg.get("retries", 3),
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        respect_retry_after_header=True,
    )
    session = requests.Session()
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers["User-Agent"] = scraper_cfg["user_agent"]
    return session


def parse_page(html: str, airport_name: str) -> list[dict[str, Any]]:
    """Every flight on one airport's page. Raises LayoutChangedError if the tables aren't recognizable.

    A small airport's board is often empty: its tables are there with headers and no rows,
    which is fine. A page without any of the four tables is a changed layout or an error page.
    """
    soup = BeautifulSoup(html, "html.parser")
    table_ids = get_app_config()["scraper"]["flight_table_ids"]
    if not any(soup.find("div", id=t) for t in table_ids):
        raise LayoutChangedError("none of the flight tables is on the page")
    rows: list[dict[str, Any]] = []
    for table_id in table_ids:
        rows.extend(scrape_table(soup, table_id, airport_name))
    return rows


def page_file_name(airport_url: str, index: int) -> str:
    """The airport's numeric id from its URL ("/102/...") as a file name; the position otherwise."""
    match = re.match(r"/?(\d+)/", airport_url)
    return f"{match.group(1) if match else index}.html.gz"


def _flights_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if not df.empty:
        departure = df["flight_type"].str.lower().str.contains("departure")
        df["destination"] = df["destination_or_origin"].where(departure)
        df["origin"] = df["destination_or_origin"].where(~departure)
    return df


def scrape(out_dir: Path | None = None, config: dict[str, Any] | None = None) -> ScrapeResult:
    """Fetch every configured airport, saving each page to `out_dir` (if given), and parse it."""
    cfg = (config or get_app_config())["scraper"]
    session = make_session(cfg)
    timeout = tuple(cfg.get("timeout_seconds", (5, 20)))
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    pages: list[PageResult] = []
    rows: list[dict[str, Any]] = []
    for i, airport in enumerate(cfg["airports"]):
        if i:
            time.sleep(cfg.get("delay_seconds", 1.5))  # be polite: one airport at a time
        url = urljoin(cfg["base_url"], airport["url"])
        page = PageResult(airport["name"], url, PageStatus.OK, fetched_at=datetime.now().isoformat(timespec="seconds"))
        try:
            response = session.get(url, timeout=timeout)
            response.raise_for_status()
        except requests.RequestException as e:
            page.status, page.error = PageStatus.FETCH_FAILED, str(e)[:300]
            logger.error("Could not fetch %s after retries: %s", airport["name"], e)
            pages.append(page)
            continue
        if out_dir is not None:
            page.file = page_file_name(airport["url"], i)
            (out_dir / page.file).write_bytes(gzip.compress(response.content))
        try:
            parsed = parse_page(response.text, airport["name"])
        except LayoutChangedError as e:
            page.status, page.error = PageStatus.LAYOUT_CHANGED, str(e)
            logger.error("Layout changed on %s: %s", airport["name"], e)
        else:
            page.rows = len(parsed)
            rows.extend(parsed)
        pages.append(page)

    if out_dir is not None:
        manifest = {"scraped_at": datetime.now().isoformat(timespec="seconds"), "pages": [asdict(p) for p in pages]}
        (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    return ScrapeResult(_flights_frame(rows), pages)


def reparse(run_dir: Path) -> ScrapeResult:
    """Parse the pages saved by an earlier `scrape` again, e.g. after a parser fix."""
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    pages, rows = [], []
    for entry in manifest["pages"]:
        page = PageResult(**entry)
        if page.file:
            html = gzip.decompress((run_dir / page.file).read_bytes()).decode("utf-8")
            try:
                parsed = parse_page(html, page.airport)
                page.status, page.rows, page.error = PageStatus.OK, len(parsed), None
                rows.extend(parsed)
            except LayoutChangedError as e:
                page.status, page.rows, page.error = PageStatus.LAYOUT_CHANGED, 0, str(e)
        pages.append(page)
    return ScrapeResult(_flights_frame(rows), pages)


def get_all_flights_data() -> pd.DataFrame:
    """Scrape every airport without keeping the pages (the `aeroscrape` command)."""
    return scrape().flights
