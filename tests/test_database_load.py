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

    assert second.inserted == 2
    with db.get_connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM departures WHERE flight_number IN ('TST010', 'TST011')")
        assert cur.fetchone()[0] == 2
