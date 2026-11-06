"""US Bureau of Transportation Statistics on-time data, mapped to the cleaned flight schema.

BTS publishes every US domestic flight by month ("Reporting Carrier On-Time Performance",
https://www.transtats.bts.gov). It includes the aircraft tail number, so the same
delay-propagation features work on it unchanged; this module only translates its columns.

Each BTS row is one flight, so it becomes two rows of the cleaned schema: a departure at
the origin and an arrival at the destination, just as the Iranian boards list a flight once
at each end. Times are local clock times at each airport, like the FIDS pages.
"""

import io
import logging
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from aeroscrape.data.clean import FLIGHT_KEY, MAX_PLAUSIBLE_DELAY, MIN_PLAUSIBLE_DELAY, CleaningAudit

logger = logging.getLogger(__name__)

URL = "https://transtats.bts.gov/PREZIP/On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{year}_{month}.zip"
USER_AGENT = "aeroscrape research client"
COLUMNS = [
    "FlightDate",
    "Reporting_Airline",
    "Tail_Number",
    "Flight_Number_Reporting_Airline",
    "Origin",
    "Dest",
    "CRSDepTime",
    "DepDelay",
    "CRSArrTime",
    "ArrDelay",
    "Cancelled",
    "Diverted",
]


def month_path(cache_dir: Path, year: int, month: int) -> Path:
    return cache_dir / f"bts_{year}_{month:02d}.zip"


def download_month(year: int, month: int, cache_dir: Path, timeout: float = 300) -> Path:
    """Download one month's zip into the cache, once. A partial download is never left behind."""
    path = month_path(cache_dir, year, month)
    if path.exists() and zipfile.is_zipfile(path):
        return path
    cache_dir.mkdir(parents=True, exist_ok=True)
    url = URL.format(year=year, month=month)
    logger.info("Downloading %s", url)
    partial = path.with_suffix(".part")
    with requests.get(url, headers={"User-Agent": USER_AGENT}, stream=True, timeout=(15, timeout)) as response:
        response.raise_for_status()
        with open(partial, "wb") as f:
            for chunk in response.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    if not zipfile.is_zipfile(partial):
        partial.unlink()
        raise RuntimeError(f"{url} did not return a zip file; BTS may not have published {year}-{month:02d} yet")
    partial.rename(path)
    return path


def read_month(path: Path, airports: list[str] | None = None) -> pd.DataFrame:
    """Read the needed columns of one month, keeping flights that touch `airports` (all if None)."""
    with zipfile.ZipFile(path) as archive:
        csv_name = next(n for n in archive.namelist() if n.lower().endswith(".csv"))
        with archive.open(csv_name) as f:
            df = pd.read_csv(io.TextIOWrapper(f, encoding="latin-1"), usecols=COLUMNS, dtype={"Tail_Number": str})
    if airports:
        df = df[df["Origin"].isin(airports) | df["Dest"].isin(airports)]
    return df


def _clock(date: pd.Series, hhmm: pd.Series) -> pd.Series:
    """Date plus an hhmm clock time; BTS writes midnight as 2400."""
    hhmm = pd.to_numeric(hhmm, errors="coerce")
    return pd.to_datetime(date) + pd.to_timedelta(hhmm // 100 * 60 + hhmm % 100, unit="m")


def to_flights(bts: pd.DataFrame) -> pd.DataFrame:
    """Departure and arrival rows in the cleaned schema (see data/clean.py)."""
    cancelled = bts["Cancelled"].fillna(0).astype(bool)
    diverted = bts["Diverted"].fillna(0).astype(bool)
    dep_sched = _clock(bts["FlightDate"], bts["CRSDepTime"])
    arr_sched = _clock(bts["FlightDate"], bts["CRSArrTime"])
    # Arrival clock times are local to the destination; one more than two hours before the
    # departure clock time means the flight lands the next day.
    arr_sched = arr_sched.where(arr_sched >= dep_sched - pd.Timedelta(hours=2), arr_sched + pd.Timedelta(days=1))
    tail = bts["Tail_Number"].where(bts["Tail_Number"].str.fullmatch(r"N?[0-9A-Z]{2,6}", na=False))
    number = bts["Reporting_Airline"] + bts["Flight_Number_Reporting_Airline"].astype(str)

    def rows(direction: str, airport: str, other: str, scheduled: pd.Series, delay: pd.Series, done: str):
        happened = delay.notna() & ~cancelled & ~diverted
        status = np.select([cancelled, diverted, happened], ["cancelled", "unknown", done], default="scheduled")
        return pd.DataFrame(
            {
                "airport": bts[airport],
                "flight_type": f"Domestic {direction.title()}",
                "direction": direction,
                "is_international": 0,
                "airline": bts["Reporting_Airline"],
                "flight_number": number,
                "destination_or_origin": bts[other],
                "register": tail,
                "aircraft": np.nan,  # not in the on-time data
                "status": status,
                "status_raw": status,
                "scheduled_datetime": scheduled,
                "actual_datetime": (scheduled + pd.to_timedelta(delay, unit="m")).where(happened),
                "actual_source": pd.Series("bts", index=bts.index).where(happened),
                "delay_minutes": delay.where(happened & delay.between(MIN_PLAUSIBLE_DELAY, MAX_PLAUSIBLE_DELAY)),
            }
        )

    flights = pd.concat(
        [
            rows("departure", "Origin", "Dest", dep_sched, bts["DepDelay"], "departed"),
            rows("arrival", "Dest", "Origin", arr_sched, bts["ArrDelay"], "landed"),
        ],
        ignore_index=True,
    )
    flights = flights.drop_duplicates(list(FLIGHT_KEY))
    return flights.sort_values("scheduled_datetime", kind="stable").reset_index(drop=True)


def load_bts_flights(
    months: list[str],
    airports: list[str] | None,
    cache_dir: Path,
    audit: CleaningAudit | None = None,
    download: bool = True,
) -> tuple[pd.DataFrame, list[Path]]:
    """Cleaned flights touching `airports` for months like "2026-03". Returns the flights and the source files."""
    audit = audit or CleaningAudit()
    paths, parts = [], []
    for month in months:
        year, mon = (int(x) for x in month.split("-"))
        path = download_month(year, mon, cache_dir) if download else month_path(cache_dir, year, mon)
        paths.append(path)
        parts.append(read_month(path, airports))
    bts = pd.concat(parts, ignore_index=True)
    audit.record(f"BTS flights touching {', '.join(airports) if airports else 'any airport'}", len(bts))
    flights = to_flights(bts)
    audit.record("Departure and arrival rows", len(flights))
    return flights, paths


def main() -> None:
    """Download the months configured in the BTS profile into its cache directory."""
    from aeroscrape.config import apply_profile, get_app_config, resolve_path

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    settings = apply_profile(get_app_config(), "bts")["data_sources"]["bts"]
    cache_dir = resolve_path(settings["cache_dir"])
    for month in settings["months"]:
        year, mon = (int(x) for x in month.split("-"))
        path = download_month(year, mon, cache_dir)
        logger.info("%s: %s (%.1f MB)", month, path, path.stat().st_size / 1e6)


if __name__ == "__main__":
    main()
