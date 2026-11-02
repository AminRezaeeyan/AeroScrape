import numpy as np
import pandas as pd
import pytest

from aeroscrape.features.build import FeatureSettings, build_features

T = pd.Timestamp


def _flight(direction: str, airport: str, sched: str, actual: str | None, register: str | None = "EPAAA", **kw):
    scheduled = T(sched)
    actual_ts = T(actual) if actual else pd.NaT
    return {
        "direction": direction,
        "flight_type": f"Domestic {'Departure' if direction == 'departure' else 'Arrival'}",
        "airport": airport,
        "airline": kw.get("airline", "ماهان"),
        "destination_or_origin": kw.get("dest", "مشهد"),
        "flight_number": kw.get("number", f"F{sched[-5:]}"),
        "register": register,
        "scheduled_datetime": scheduled,
        "actual_datetime": actual_ts,
        "delay_minutes": (actual_ts - scheduled).total_seconds() / 60 if actual else np.nan,
    }


@pytest.fixture
def context() -> pd.DataFrame:
    return pd.DataFrame(
        [
            _flight("departure", "MHD", "2025-06-01 06:00", "2025-06-01 06:30"),  # earlier leg, 30 min late
            _flight("arrival", "THR", "2025-06-01 08:00", "2025-06-01 08:40"),  # inbound, 40 min late
            _flight("departure", "THR", "2025-06-01 09:30", "2025-06-01 10:10"),  # the target
            _flight("departure", "THR", "2025-06-01 09:45", "2025-06-01 09:50", register=None, number="X1"),
        ]
    )


def _target(context: pd.DataFrame, lead: int = 60) -> pd.Series:
    targets = context[context["flight_number"] == "F09:30"]
    return build_features(targets, context, FeatureSettings(prediction_lead_minutes=lead)).iloc[0]


def test_schedule_rotation_features(context: pd.DataFrame) -> None:
    row = _target(context)
    assert row["has_inbound"] == 1
    assert row["scheduled_turnaround_min"] == 90
    assert row["nth_flight_of_day_for_aircraft"] == 2


def test_inbound_delay_is_hidden_until_it_has_landed(context: pd.DataFrame) -> None:
    # At 08:30 (one hour before departure) the inbound hadn't landed: its delay is unknown,
    # but it is known to be at least 30 minutes overdue.
    row = _target(context, lead=60)
    assert np.isnan(row["inbound_delay_min"])
    assert row["inbound_overdue_min"] == 30

    # At 09:00 it had landed at 08:40, 40 minutes late
    row = _target(context, lead=30)
    assert row["inbound_delay_min"] == 40
    assert np.isnan(row["inbound_overdue_min"])


def test_previous_departure_of_the_same_aircraft(context: pd.DataFrame) -> None:
    assert _target(context)["prev_departure_delay_min"] == 30


def test_flight_without_registration_has_no_rotation_features(context: pd.DataFrame) -> None:
    row = build_features(context[context["flight_number"] == "X1"], context).iloc[0]
    assert row["has_inbound"] == 0
    assert np.isnan(row["scheduled_turnaround_min"])
    assert np.isnan(row["prev_departure_delay_min"])


def test_inbound_before_midnight_is_linked() -> None:
    context = pd.DataFrame(
        [
            _flight("arrival", "THR", "2025-06-01 23:50", "2025-06-01 23:55"),
            _flight("departure", "THR", "2025-06-02 00:30", "2025-06-02 00:40"),
        ]
    )
    row = build_features(context[context["direction"] == "departure"], context).iloc[0]
    assert row["scheduled_turnaround_min"] == 40


def test_congestion_counts(context: pd.DataFrame) -> None:
    row = _target(context)
    assert row["departures_same_hour"] == 2  # 09:30 and 09:45 at THR
    assert row["airline_departures_today"] == 3


def _history() -> pd.DataFrame:
    rows = []
    for day in range(1, 11):
        for i in range(5):
            sched = f"2025-06-{day:02d} {8 + i:02d}:00"
            actual = str(T(sched) + pd.Timedelta(minutes=5 * day + i))
            rows.append(_flight("departure", "THR", sched, actual, register=None, number=f"D{day}{i}"))
    return pd.DataFrame(rows)


def test_rates_never_see_the_flights_own_day_or_later() -> None:
    context = _history()
    targets = context[context["scheduled_datetime"].dt.day == 8]
    before = build_features(targets, context)

    # Change every outcome from June 8 onward; rates for June 8 must not move
    changed = context.copy()
    future = changed["scheduled_datetime"].dt.day >= 8
    changed.loc[future, "delay_minutes"] = changed.loc[future, "delay_minutes"] * 10 + 100
    after = build_features(targets, changed)

    for col in ("airline_delay_rate_7d", "route_median_delay_7d"):
        pd.testing.assert_series_equal(before[col], after[col])


def test_rates_do_change_when_the_past_changes() -> None:
    context = _history()
    targets = context[context["scheduled_datetime"].dt.day == 8]
    changed = context.copy()
    past = changed["scheduled_datetime"].dt.day < 8
    changed.loc[past, "delay_minutes"] += 200

    before = build_features(targets, context)["route_median_delay_7d"]
    after = build_features(targets, changed)["route_median_delay_7d"]
    assert (after > before).all()


def test_first_day_has_no_rate() -> None:
    context = _history()
    first = build_features(context[context["scheduled_datetime"].dt.day == 1], context)
    assert first["airline_delay_rate_7d"].isna().all()
