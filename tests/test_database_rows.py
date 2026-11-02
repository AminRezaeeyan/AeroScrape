from datetime import datetime

import pandas as pd
import pytest

from aeroscrape.storage.database import Database

SUNDAY = datetime(2026, 11, 1)


@pytest.fixture
def db() -> Database:
    # Row preparation and time parsing need no connection
    return Database.__new__(Database)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("دوشنبه 06:20", datetime(2026, 11, 2, 6, 20)),  # spelling used by the site
        ("دو شنبه 06:20", datetime(2026, 11, 2, 6, 20)),
        ("سه‌شنبه 23:10", datetime(2026, 11, 3, 23, 10)),  # zero-width non-joiner
        ("يكشنبه 07:00", datetime(2026, 11, 1, 7, 0)),  # Arabic ي and ك
        ("شنبه 10:00", datetime(2026, 10, 31, 10, 0)),
        ("2025-06-01 00:05:00", datetime(2025, 6, 1, 0, 5)),  # Gregorian, as written by export_csv
        ("1404-08-11 06:20", datetime(2025, 11, 2, 6, 20)),  # Persian calendar
    ],
)
def test_scheduled_time_parsing(db: Database, text: str, expected: datetime) -> None:
    assert db._parse_time_with_weekday(text, SUNDAY)[0] == expected


def _row(**overrides: str) -> pd.Series:
    row = {
        "scheduled_time": "دوشنبه 06:20",
        "airline": "ماهان",
        "flight_number": "IRM1036",
        "destination": "تهران",
        "status": "پروازکرد06:35",
        "actual_time": "06:35",
        "airport": "فرودگاه مشهد",
        "flight_type": "Domestic Departure",
    }
    return pd.Series({**row, **overrides})


def test_valid_departure_is_prepared(db: Database) -> None:
    table, values = db._prepare_flight(_row(), SUNDAY)

    assert table == "departures"
    assert values["place"] == "تهران"
    assert values["actual_time"] == datetime(2026, 11, 2, 6, 35)


def test_after_midnight_actual_time_moves_to_next_day(db: Database) -> None:
    _, values = db._prepare_flight(_row(scheduled_time="دوشنبه 23:50", actual_time="00:20"), SUNDAY)
    assert values["actual_time"] == datetime(2026, 11, 3, 0, 20)


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"airline": ""}, "missing airline"),
        ({"flight_number": " "}, "missing flight_number"),
        ({"destination": ""}, "missing place"),
        ({"scheduled_time": ""}, "missing scheduled_time"),
        ({"scheduled_time": "soon"}, "unparseable scheduled_time"),
    ],
)
def test_incomplete_rows_are_skipped_with_a_reason(db: Database, overrides: dict[str, str], reason: str) -> None:
    assert db._prepare_flight(_row(**overrides), SUNDAY) == reason
