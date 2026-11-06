"""The API's prediction log: one row per scored flight, joined to the actual delay later.

A flight is identified by airport, flight number and scheduled time (`flight_key`), in the
cleaner's normalized form, so a logged prediction and the scraped flight compare equal.
Requests without a flight number can't be matched to an outcome; they still count for drift.
"""

import json
import logging
from dataclasses import dataclass
from typing import Any

import pandas as pd
from psycopg2.extras import execute_values

from aeroscrape.data.clean import normalize_persian

logger = logging.getLogger(__name__)

TRAFFIC_SOURCES = ("client", "replay", "loadtest")

_INSERT = """
    INSERT INTO predictions (
        request_id, traffic_source, flight_key, airport, airline, flight_number, scheduled_time,
        model_version, p_delay, likely_delayed, q10, q50, q90, features
    ) VALUES %s
"""
_TEMPLATE = "(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)"


def flight_key(airport: Any, flight_number: Any, scheduled: Any) -> str | None:
    if flight_number is None or (isinstance(flight_number, float) and pd.isna(flight_number)):
        return None
    number = normalize_persian(str(flight_number))
    if not number:
        return None
    return f"{normalize_persian(str(airport))}|{number}|{pd.Timestamp(scheduled):%Y-%m-%d %H:%M}"


def _json_value(value: Any) -> Any:
    if hasattr(value, "item"):  # numpy scalars
        return value.item()
    return str(value)


@dataclass(frozen=True)
class PredictionRecord:
    request_id: str
    traffic_source: str
    flight_key: str | None
    airport: str
    airline: str
    flight_number: str | None
    scheduled_time: pd.Timestamp
    model_version: str
    p_delay: float
    likely_delayed: bool
    q10: float
    q50: float
    q90: float
    features: str  # JSON

    def as_row(self) -> tuple:
        return (
            self.request_id,
            self.traffic_source,
            self.flight_key,
            self.airport,
            self.airline,
            self.flight_number,
            self.scheduled_time.to_pydatetime(),
            self.model_version,
            self.p_delay,
            self.likely_delayed,
            self.q10,
            self.q50,
            self.q90,
            self.features,
        )


def build_records(
    request_id: str,
    traffic_source: str,
    flights: list[dict[str, Any]],
    X: pd.DataFrame,
    out: pd.DataFrame,
    model_version: str,
) -> list[PredictionRecord]:
    """Records for scored flights: `flights` are the request dicts, `X` the model input, `out` its output."""
    features = X.astype(object).where(X.notna(), None).to_dict("records")
    records = []
    for flight, row_features, (_, row) in zip(flights, features, out.iterrows(), strict=True):
        scheduled = pd.Timestamp(flight["scheduled_datetime"])
        airport = str(normalize_persian(flight["airport"]))
        records.append(
            PredictionRecord(
                request_id=request_id,
                traffic_source=traffic_source,
                flight_key=flight_key(airport, flight.get("flight_number"), scheduled),
                airport=airport,
                airline=str(normalize_persian(flight["airline"])),
                flight_number=flight.get("flight_number"),
                scheduled_time=scheduled,
                model_version=model_version,
                p_delay=float(row["p_delay_over_15"]),
                likely_delayed=bool(row["likely_delayed"]),
                q10=float(row["p10"]),
                q50=float(row["p50"]),
                q90=float(row["p90"]),
                features=json.dumps(row_features, ensure_ascii=False, default=_json_value),
            )
        )
    return records


class PredictionLog:
    """Writes records with one multi-row INSERT. `source` is anything with get_connection()."""

    def __init__(self, source: Any):
        self.source = source

    def write(self, records: list[PredictionRecord]) -> None:
        if not records:
            return
        conn = self.source.get_connection()
        try:
            with conn, conn.cursor() as cur:
                execute_values(cur, _INSERT, [r.as_row() for r in records], template=_TEMPLATE)
        finally:
            conn.close()
