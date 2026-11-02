import pandas as pd
import pytest

from aeroscrape.data.clean import (
    CleaningAudit,
    FlightStatus,
    clean_flights,
    normalize_persian,
    parse_status,
    select_training_rows,
)


@pytest.mark.parametrize(
    ("text", "status", "time"),
    [
        ("پروازکرد04:57", FlightStatus.DEPARTED, "04:57"),  # current site
        ("پرواز كرد", FlightStatus.DEPARTED, None),  # 2025 spelling, Arabic ك
        ("تخمین نشست 07:10 | نشست07:15 | پایان دریافت بار", FlightStatus.LANDED, "07:15"),
        ("تخمین نشست 11:45 | طبق برنامه", FlightStatus.SCHEDULED, None),  # an estimate is not a landing
        ("تاخیردارد00:35", FlightStatus.DELAYED, None),  # an expected time is not an actual time
        ("باطل شد", FlightStatus.CANCELLED, None),
        ("", FlightStatus.UNKNOWN, None),
    ],
)
def test_parse_status(text: str, status: FlightStatus, time: str | None) -> None:
    assert parse_status(text) == (status, time)


def test_normalize_persian_unifies_letters_and_spaces() -> None:
    assert normalize_persian(" فرودگاه  شيراز ") == "فرودگاه شیراز"
    assert normalize_persian("كيش") == normalize_persian("کیش")


def _raw(**overrides: str) -> dict[str, str | None]:
    row: dict[str, str | None] = {
        "scheduled_time": "2025-06-01 10:00:00",
        "airline": "ماهان",
        "flight_number": "IRM1",
        "destination_or_origin": "مشهد",
        "status": "پرواز كرد",
        "counter": None,
        "actual_time": "2025-06-01 10:20:00",
        "register": "EPMOH",
        "aircraft": "A310",
        "airport": "فرودگاه مهرآباد",
        "flight_type": "Domestic Departure",
    }
    return {**row, **overrides}


def test_status_time_wins_over_the_actual_time_cell() -> None:
    flights = clean_flights(pd.DataFrame([_raw(status="پروازکرد10:30")]))
    assert flights.loc[0, "delay_minutes"] == 30
    assert flights.loc[0, "actual_source"] == "status"


def test_actual_time_is_ignored_unless_the_flight_happened() -> None:
    flights = clean_flights(pd.DataFrame([_raw(status="تاخیردارد")]))
    assert pd.isna(flights.loc[0, "actual_datetime"])


def test_event_time_after_midnight_goes_to_the_next_day() -> None:
    flights = clean_flights(pd.DataFrame([_raw(scheduled_time="2025-06-01 23:50:00", status="پروازکرد00:20")]))
    assert flights.loc[0, "actual_datetime"] == pd.Timestamp("2025-06-02 00:20")
    assert flights.loc[0, "delay_minutes"] == 30


def test_repeated_scrapes_keep_the_most_advanced_status() -> None:
    flights = clean_flights(pd.DataFrame([_raw(status="طبق برنامه", actual_time=None), _raw()]))
    assert len(flights) == 1
    assert flights.loc[0, "status"] == "departed"


def test_training_rows_are_departed_domestic_departures_and_the_audit_counts_them() -> None:
    raw = pd.DataFrame(
        [
            _raw(),
            _raw(flight_number="IRM2", status="باطل شد", actual_time=None),
            _raw(flight_number="IRM3", flight_type="Domestic Arrival", status="نشست"),
        ]
    )
    audit = CleaningAudit()
    rows = select_training_rows(clean_flights(raw, audit), audit)

    assert rows["flight_number"].tolist() == ["IRM1"]
    assert dict(audit.steps)["Domestic departures"] == 2
