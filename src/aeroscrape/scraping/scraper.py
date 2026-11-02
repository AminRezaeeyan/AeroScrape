import logging
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup

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


def scrape_airport_data(airport_name, airport_url):
    app_config = get_app_config()
    base_url = app_config["scraper"]["base_url"]
    user_agent = app_config["scraper"]["user_agent"]

    headers = {"User-Agent": user_agent}

    full_url = urljoin(base_url, airport_url)
    try:
        response = requests.get(full_url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        all_flights = []
        for table_id in app_config["scraper"]["flight_table_ids"]:
            try:
                flights = scrape_table(soup, table_id, airport_name)
            except LayoutChangedError as e:
                logger.error("Layout changed on %s, table %r: %s", airport_name, table_id, e)
                continue
            all_flights.extend(flights)
        return all_flights
    except requests.RequestException as e:
        logger.error("Error fetching data for %s: %s", airport_name, e)
        return []


def get_all_flights_data() -> pd.DataFrame:
    app_config = get_app_config()
    airports_config = app_config["scraper"]["airports"]

    all_data = []
    for airport in airports_config:
        logger.info("Scraping data for %s", airport["name"])
        airport_data = scrape_airport_data(airport["name"], airport["url"])
        all_data.extend(airport_data)

    df = pd.DataFrame(all_data)

    if not df.empty:
        df["destination"] = df.apply(
            lambda row: row["destination_or_origin"] if "departure" in row["flight_type"].lower() else None, axis=1
        )
        df["origin"] = df.apply(
            lambda row: row["destination_or_origin"] if "arrival" in row["flight_type"].lower() else None, axis=1
        )

    return df
