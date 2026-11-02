"""Turn scraped flight rows into one tidy table of all flights.

Every flight type is kept: departures are what we predict, and arrivals are
needed as a feature source (when did this aircraft land before its next
departure?). Choosing training rows happens later, in `select_training_rows`,
and every filter there is counted so data loss is visible.
"""

import logging
import re
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Delays outside this range are scraping or parsing errors, not real delays
MIN_PLAUSIBLE_DELAY = -60
MAX_PLAUSIBLE_DELAY = 12 * 60

TEXT_COLUMNS = ("airline", "flight_number", "destination_or_origin", "status", "register", "aircraft", "airport")
FLIGHT_KEY = ("airport", "flight_type", "flight_number", "scheduled_datetime")

_PERSIAN_FIXES = str.maketrans({"ي": "ی", "ك": "ک", "‌": " ", "‏": None, "‎": None})
_TIME = re.compile(r"(\d{1,2}):(\d{2})")


def normalize_persian(text: object) -> object:
    """Unify Arabic and Persian letters and whitespace so one airline isn't two categories."""
    if not isinstance(text, str):
        return text
    cleaned = " ".join(text.translate(_PERSIAN_FIXES).split())
    return cleaned or None


class FlightStatus(StrEnum):
    CANCELLED = "cancelled"
    DEPARTED = "departed"
    LANDED = "landed"
    APPROACHING = "approaching"
    BOARDING = "boarding"
    CHECK_IN = "check_in"
    DELAYED = "delayed"
    RESCHEDULED = "rescheduled"
    SCHEDULED = "scheduled"
    NO_PASSENGERS = "no_passengers"
    UNKNOWN = "unknown"


# Keyword (compared without spaces) -> status. A status cell can list several states,
# e.g. "تخمین نشست 07:10 | نشست07:15 | پایان دریافت بار"; the most advanced one wins,
# in the order of this list.
STATUS_KEYWORDS: list[tuple[str, FlightStatus]] = [
    ("باطل", FlightStatus.CANCELLED),
    ("پروازکرد", FlightStatus.DEPARTED),
    ("نشست", FlightStatus.LANDED),
    ("پایاندریافتبار", FlightStatus.LANDED),
    ("دریافتبار", FlightStatus.LANDED),
    ("اعلامورود", FlightStatus.APPROACHING),
    ("سوار", FlightStatus.BOARDING),
    ("آمادهپرواز", FlightStatus.BOARDING),
    ("دریافتکارتپرواز", FlightStatus.BOARDING),
    ("پذیرش", FlightStatus.CHECK_IN),
    ("سالنبازرسی", FlightStatus.CHECK_IN),
    ("تاخیر", FlightStatus.DELAYED),
    ("زمانجدید", FlightStatus.RESCHEDULED),
    ("طبقبرنامه", FlightStatus.SCHEDULED),
    ("بدونمسافر", FlightStatus.NO_PASSENGERS),
]
_PRECEDENCE = {status: rank for rank, status in enumerate(dict.fromkeys(s for _, s in STATUS_KEYWORDS))}
# Only these states carry the time the event actually happened
_EVENT_STATUSES = {FlightStatus.DEPARTED: "پروازکرد", FlightStatus.LANDED: "نشست"}


def parse_status(text: object) -> tuple[FlightStatus, str | None]:
    """Return the most advanced status in a status cell and the event time it states, if any.

    "تخمین نشست" is an *estimated* landing, so it is removed before matching "نشست".
    """
    if not isinstance(text, str) or not text.strip():
        return FlightStatus.UNKNOWN, None
    normalized = normalize_persian(text)
    if not isinstance(normalized, str):
        return FlightStatus.UNKNOWN, None
    parts = [p.replace(" ", "").replace("تخمیننشست", "تخمین") for p in normalized.split("|")]
    best, event_time = FlightStatus.UNKNOWN, None
    for part in parts:
        for keyword, status in STATUS_KEYWORDS:
            if keyword in part:
                if best is FlightStatus.UNKNOWN or _PRECEDENCE[status] < _PRECEDENCE[best]:
                    best = status
                break
    event_keyword = _EVENT_STATUSES.get(best)
    if event_keyword:
        for part in parts:
            match = _TIME.search(part) if event_keyword in part else None
            if match:
                event_time = f"{int(match.group(1)):02d}:{match.group(2)}"
                break
    return best, event_time


def _closest_datetime(hhmm: str, reference: pd.Timestamp) -> pd.Timestamp:
    """Put an HH:MM time on the day (reference's day, or the one before or after) closest to reference.

    This assumes events happen within 12 hours of the schedule, which holds for all but a handful of flights.
    """
    hours, minutes = map(int, hhmm.split(":"))
    base = reference.normalize() + pd.Timedelta(hours=hours, minutes=minutes)
    candidates = [base + pd.Timedelta(days=d) for d in (-1, 0, 1)]
    return min(candidates, key=lambda c: abs(c - reference))


@dataclass
class CleaningAudit:
    """Row counts after each step, so it is clear where data is lost."""

    steps: list[tuple[str, int]] = field(default_factory=list)

    def record(self, step: str, rows: int) -> None:
        self.steps.append((step, rows))
        logger.info("%-55s %6d rows", step, rows)

    def to_markdown(self) -> str:
        lines = ["| Step | Rows |", "| --- | --- |"]
        lines += [f"| {step} | {rows:,} |" for step, rows in self.steps]
        return "\n".join(lines)


def clean_flights(raw: pd.DataFrame, audit: CleaningAudit | None = None) -> pd.DataFrame:
    """Normalize every scraped flight. Returns one row per flight with parsed times and status."""
    audit = audit or CleaningAudit()
    df = raw.copy()
    audit.record("Raw scraped rows", len(df))

    for col in TEXT_COLUMNS:
        if col in df.columns:
            df[col] = df[col].map(normalize_persian)

    df["scheduled_datetime"] = pd.to_datetime(df["scheduled_time"], errors="coerce")
    df["actual_time_cell"] = pd.to_datetime(df["actual_time"], errors="coerce")
    df = df[df["scheduled_datetime"].notna() & df["flight_number"].notna()]
    audit.record("With a scheduled time and flight number", len(df))

    # A flight seen in several scrapes keeps its most advanced state (the latest status wins on ties)
    parsed = df["status"].map(parse_status)
    df["status_raw"] = df["status"]
    df["status"] = [s.value for s, _ in parsed]
    df["status_event_time"] = [t for _, t in parsed]
    df["_rank"] = df["status"].map({s.value: r for s, r in _PRECEDENCE.items()}).fillna(len(_PRECEDENCE))
    df = df.sort_values("_rank").drop_duplicates(list(FLIGHT_KEY), keep="first").drop(columns="_rank")
    audit.record("After removing duplicate flights", len(df))

    df["direction"] = np.where(df["flight_type"].str.contains("Departure"), "departure", "arrival")
    df["is_international"] = df["flight_type"].str.startswith("International").astype(int)

    # Only a departed or landed flight has an actual time. For other statuses the "actual time"
    # cell holds an expected time. The status text states when the event really happened, and
    # the cell sometimes holds an estimate even then, so it is only the fallback. Dates stored
    # by older scraper versions can be a day off, so the HH:MM is placed on the day closest to
    # the schedule.
    happened = df["status"].isin([s.value for s in _EVENT_STATUSES])
    cell_hhmm = df["actual_time_cell"].dt.strftime("%H:%M")
    actual: list[pd.Timestamp | None] = []
    source: list[str | None] = []
    for occurred, status_hhmm, hhmm_cell, scheduled in zip(
        happened, df["status_event_time"], cell_hhmm, df["scheduled_datetime"], strict=True
    ):
        if occurred and status_hhmm:
            actual.append(_closest_datetime(status_hhmm, scheduled))
            source.append("status")
        elif occurred and isinstance(hhmm_cell, str):
            actual.append(_closest_datetime(hhmm_cell, scheduled))
            source.append("actual_time")
        else:
            actual.append(None)
            source.append(None)
    df["actual_datetime"] = pd.to_datetime(pd.Series(actual, index=df.index))
    df["actual_source"] = source

    delay = (df["actual_datetime"] - df["scheduled_datetime"]).dt.total_seconds() / 60
    implausible = delay.notna() & ~delay.between(MIN_PLAUSIBLE_DELAY, MAX_PLAUSIBLE_DELAY)
    if implausible.any():
        logger.warning(
            "Ignoring %d implausible delays outside [%d, %d] min",
            implausible.sum(),
            MIN_PLAUSIBLE_DELAY,
            MAX_PLAUSIBLE_DELAY,
        )
    df["delay_minutes"] = delay.mask(implausible)

    columns = [
        "airport",
        "flight_type",
        "direction",
        "is_international",
        "airline",
        "flight_number",
        "destination_or_origin",
        "register",
        "aircraft",
        "status",
        "status_raw",
        "scheduled_datetime",
        "actual_datetime",
        "actual_source",
        "delay_minutes",
    ]
    return df[columns].sort_values("scheduled_datetime", kind="stable").reset_index(drop=True)


def select_training_rows(flights: pd.DataFrame, audit: CleaningAudit | None = None) -> pd.DataFrame:
    """Domestic departures that departed with a known, plausible delay: the rows we learn from."""
    audit = audit or CleaningAudit()
    rows = flights[flights["flight_type"] == "Domestic Departure"]
    audit.record("Domestic departures", len(rows))
    rows = rows[rows["status"] == FlightStatus.DEPARTED.value]
    audit.record("... with status departed", len(rows))
    rows = rows[rows["actual_datetime"].notna()]
    audit.record("... with a departure time", len(rows))
    rows = rows[rows["delay_minutes"].notna()]
    audit.record(f"... with a delay in [{MIN_PLAUSIBLE_DELAY}, {MAX_PLAUSIBLE_DELAY}] min", len(rows))
    return rows.reset_index(drop=True)


def read_flights(path: str | Path) -> pd.DataFrame:
    """Read a cleaned flights file with its datetime columns parsed."""
    return pd.read_csv(path, parse_dates=["scheduled_datetime", "actual_datetime"])


def clean_flights_file(raw_path: str | Path, cleaned_path: str | Path) -> CleaningAudit:
    logger.info("Cleaning raw flights from %s", raw_path)
    audit = CleaningAudit()
    flights = clean_flights(pd.read_csv(raw_path, encoding="utf-8-sig", dtype=str), audit)
    select_training_rows(flights, audit)  # recorded for the audit; the file keeps every flight
    flights.to_csv(cleaned_path, index=False, encoding="utf-8-sig")
    logger.info("Wrote %d cleaned flights to %s", len(flights), cleaned_path)
    return audit
