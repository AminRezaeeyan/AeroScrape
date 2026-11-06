"""Loads flights into a real Postgres. Runs only when AEROSCRAPE_TEST_DB=1 and DB_* point at a disposable database."""

import os
from datetime import datetime

import pandas as pd
import pytest

from aeroscrape.storage.database import Database

pytestmark = pytest.mark.skipif(os.getenv("AEROSCRAPE_TEST_DB") != "1", reason="needs a disposable Postgres")


def _departure(flight_number: str, **overrides: str) -> dict[str, str]:
    row = {
        "scheduled_time": "دوشنبه 06:20",
        "airline": "ماهان",
        "flight_number": flight_number,
        "destination": "تهران",
        "status": "طبق برنامه",
        "airport": "فرودگاه تست",
        "flight_type": "Domestic Departure",
    }
    return {**row, **overrides}


def test_one_bad_row_does_not_lose_the_rest() -> None:
    db = Database()
    df = pd.DataFrame(
        [
            _departure("TST001"),
            _departure("TST002", airline=""),  # skipped before reaching the database
            _departure("TST003", counter="x" * 80),  # longer than VARCHAR(50): rejected by the database
            _departure("TST004"),
        ]
    )

    result = db.insert_flights_from_dataframe(df, base_date=datetime(2026, 11, 1))

    assert (result.inserted, result.failed, dict(result.skipped)) == (2, 1, {"missing airline": 1})
    with db.get_connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT flight_number FROM departures WHERE flight_number LIKE 'TST%' ORDER BY 1")
        assert [r[0] for r in cur.fetchall()] == ["TST001", "TST004"]


def test_loading_the_same_flights_twice_does_not_duplicate() -> None:
    db = Database()
    df = pd.DataFrame([_departure("TST010"), _departure("TST011")])

    db.insert_flights_from_dataframe(df, base_date=datetime(2026, 11, 1))
    second = db.insert_flights_from_dataframe(df, base_date=datetime(2026, 11, 1))

    assert (second.inserted, second.unchanged) == (0, 2)  # the same scrape again changes nothing
    with db.get_connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM departures WHERE flight_number IN ('TST010', 'TST011')")
        assert cur.fetchone()[0] == 2


def test_fetch_flights_returns_the_scraped_format_for_a_time_window() -> None:
    db = Database()
    db.insert_flights_from_dataframe(
        pd.DataFrame([_departure("TST020", scheduled_time="سه شنبه 07:00"), _departure("TST021")]),
        base_date=datetime(2026, 11, 1),
    )

    window = db.fetch_flights(datetime(2026, 11, 3), datetime(2026, 11, 4))  # Tuesday only

    tuesday = window[window["flight_number"].str.startswith("TST02")]
    assert tuesday["flight_number"].tolist() == ["TST020"]
    assert tuesday.iloc[0]["scheduled_time"] == "2026-11-03 07:00:00"
    assert list(window.columns) == [
        "scheduled_time",
        "airline",
        "flight_number",
        "destination_or_origin",
        "status",
        "counter",
        "actual_time",
        "register",
        "aircraft",
        "airport",
        "flight_type",
    ]


def test_a_changed_status_updates_the_flight_and_its_timestamp() -> None:
    db = Database()
    db.insert_flights_from_dataframe(pd.DataFrame([_departure("TST030")]), base_date=datetime(2026, 11, 1))
    with db.get_connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT updated_at FROM departures WHERE flight_number = 'TST030'")
        first = cur.fetchone()[0]

    changed = db.insert_flights_from_dataframe(
        pd.DataFrame([_departure("TST030", status="پرواز کرد 06:41")]), base_date=datetime(2026, 11, 1)
    )

    assert (changed.inserted, changed.unchanged) == (1, 0)
    with db.get_connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT status, updated_at FROM departures WHERE flight_number = 'TST030'")
        status, updated = cur.fetchone()
    assert status == "پرواز کرد 06:41" and updated > first
