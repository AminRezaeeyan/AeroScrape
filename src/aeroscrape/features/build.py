"""The one place features are computed, for training and (from Phase 4) for serving.

`build_features(targets, context)` adds feature columns to the departures in
`targets`, looking up related flights in `context` (all known flights, arrivals
included). Every feature uses only information available when the prediction
is made:

- schedule features (calendar, rotation, congestion) are known when the flight
  is booked;
- historical rates use only flights from earlier days;
- live propagation features use the inbound aircraft's actual times only if
  they were known `prediction_lead_minutes` before departure.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

DELAYED_THRESHOLD_MINUTES = 15
WEEKEND_DAYS = ("Thursday", "Friday")  # the Iranian weekend


@dataclass(frozen=True)
class FeatureSettings:
    prediction_lead_minutes: int = 60
    inbound_window_hours: int = 6
    previous_departure_window_hours: int = 12
    rate_window_days: int = 7
    rate_smoothing: int = 20

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "FeatureSettings":
        keys = cls.__dataclass_fields__
        return cls(**{k: v for k, v in config.get("features", {}).items() if k in keys})


# ---------------------------------------------------------------------------- calendar


def _time_of_day(hour: int) -> str:
    if 5 <= hour <= 11:
        return "Morning"
    if 12 <= hour <= 16:
        return "Afternoon"
    if 17 <= hour <= 20:
        return "Evening"
    return "Night"


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    scheduled = pd.to_datetime(df["scheduled_datetime"])
    out = df.assign(
        Scheduled_Hour_of_Day=scheduled.dt.hour,
        scheduled_minute=scheduled.dt.minute,
        scheduled_day_of_week=scheduled.dt.day_name(),
        scheduled_time_of_day=scheduled.dt.hour.map(_time_of_day),
    )
    out["is_weekend"] = out["scheduled_day_of_week"].isin(WEEKEND_DAYS).astype(int)
    return out


# Not used by the models: in a three-week window these are constant (one season, no
# Nowruz) or nearly so, so a model can't learn from them. They become useful once the
# data spans a year; enable them in config.yaml then. The holidays are Gregorian
# approximations; lunar holidays need a Hijri calendar.
def add_seasonal_features(df: pd.DataFrame) -> pd.DataFrame:
    scheduled = pd.to_datetime(df["scheduled_datetime"])
    month, day = scheduled.dt.month, scheduled.dt.day
    nowruz_13 = ((month == 3) & (day >= 21)) | ((month == 4) & (day <= 2))
    season = np.select(
        [
            nowruz_13 | month.isin([4, 5]) | ((month == 6) & (day <= 20)),
            ((month == 6) & (day >= 21)) | month.isin([7, 8]) | ((month == 9) & (day <= 22)),
            ((month == 9) & (day >= 23)) | month.isin([10, 11]) | ((month == 12) & (day <= 21)),
        ],
        ["Spring", "Summer", "Autumn"],
        default="Winter",
    )
    fixed_holidays = {(3, 20), (2, 11), (6, 4), (6, 5)}
    return df.assign(
        scheduled_season=season,
        is_Nourooz_4=((month == 3) & day.between(21, 24)).astype(int),
        is_Nourooz_13=nowruz_13.astype(int),
        Normal_holiday=(
            ~nowruz_13 & pd.Series(list(zip(month, day, strict=True)), index=df.index).isin(fixed_holidays)
        ).astype(int),
    )


# ---------------------------------------------------------------------------- rotation and propagation


def _registered(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["register"].notna()]


def add_rotation_features(targets: pd.DataFrame, context: pd.DataFrame, settings: FeatureSettings) -> pd.DataFrame:
    """Link each departure to the same aircraft's inbound arrival and earlier departures.

    A late aircraft stays late: if it landed late, its next departure is likely late too.
    Only departures with an aircraft registration can be linked.
    """
    out = targets.copy()
    out["_row"] = np.arange(len(out))
    cutoff = out["scheduled_datetime"] - pd.Timedelta(minutes=settings.prediction_lead_minutes)
    out["_cutoff"] = cutoff

    # Inbound: the same aircraft's latest scheduled arrival at this airport in the preceding window
    arrivals = _registered(context[context["direction"] == "arrival"])[
        ["register", "airport", "scheduled_datetime", "actual_datetime"]
    ].rename(columns={"scheduled_datetime": "inbound_sched", "actual_datetime": "inbound_actual"})
    linked = pd.merge_asof(
        _registered(out).sort_values("scheduled_datetime"),
        arrivals.dropna(subset=["inbound_sched"]).sort_values("inbound_sched"),
        left_on="scheduled_datetime",
        right_on="inbound_sched",
        by=["register", "airport"],
        direction="backward",
        allow_exact_matches=False,
        tolerance=pd.Timedelta(hours=settings.inbound_window_hours),
    ).set_index("_row")

    has_inbound = linked["inbound_sched"].notna()
    landed_by_cutoff = linked["inbound_actual"].notna() & (linked["inbound_actual"] <= linked["_cutoff"])
    still_in_air = linked["inbound_actual"].notna() & (linked["inbound_actual"] > linked["_cutoff"])
    inbound_delay = (linked["inbound_actual"] - linked["inbound_sched"]).dt.total_seconds() / 60
    overdue = ((linked["_cutoff"] - linked["inbound_sched"]).dt.total_seconds() / 60).clip(lower=0)

    out["has_inbound"] = out["_row"].map(has_inbound.astype(int)).fillna(0).astype(int)
    # Known in advance from the schedule
    out["scheduled_turnaround_min"] = out["_row"].map(
        ((linked["scheduled_datetime"] - linked["inbound_sched"]).dt.total_seconds() / 60).where(has_inbound)
    )
    # Known only once the inbound has landed, so only if it landed before the prediction time
    out["inbound_delay_min"] = out["_row"].map(inbound_delay.where(landed_by_cutoff))
    # If it hadn't landed by then, we still know it is at least this late
    out["inbound_overdue_min"] = out["_row"].map(overdue.where(still_in_air))

    # The same aircraft's previous departure from any airport, if its delay was known in time
    departures = _registered(context[context["direction"] == "departure"])[
        ["register", "scheduled_datetime", "actual_datetime", "delay_minutes"]
    ].rename(
        columns={"scheduled_datetime": "prev_sched", "actual_datetime": "prev_actual", "delay_minutes": "prev_delay"}
    )
    previous = pd.merge_asof(
        _registered(out).sort_values("scheduled_datetime")[["_row", "register", "scheduled_datetime", "_cutoff"]],
        departures.sort_values("prev_sched"),
        left_on="scheduled_datetime",
        right_on="prev_sched",
        by="register",
        direction="backward",
        allow_exact_matches=False,
        tolerance=pd.Timedelta(hours=settings.previous_departure_window_hours),
    ).set_index("_row")
    known = previous["prev_actual"].notna() & (previous["prev_actual"] <= previous["_cutoff"])
    out["prev_departure_delay_min"] = out["_row"].map(previous["prev_delay"].where(known))

    # Later rotations of the day tend to accumulate delay
    day_departures = _registered(context[context["direction"] == "departure"])[["register", "scheduled_datetime"]]
    day_departures = day_departures.assign(_date=day_departures["scheduled_datetime"].dt.normalize())
    day_departures = day_departures.drop_duplicates().sort_values("scheduled_datetime")
    day_departures["nth"] = day_departures.groupby(["register", "_date"]).cumcount() + 1
    nth = out.assign(_date=out["scheduled_datetime"].dt.normalize()).merge(
        day_departures, on=["register", "_date", "scheduled_datetime"], how="left"
    )["nth"]
    out["nth_flight_of_day_for_aircraft"] = nth.to_numpy()

    return out.drop(columns=["_row", "_cutoff"])


# ---------------------------------------------------------------------------- congestion


def add_congestion_features(targets: pd.DataFrame, context: pd.DataFrame) -> pd.DataFrame:
    """Scheduled traffic, known in advance: busy airports and busy airline days run later."""
    departures = context[context["direction"] == "departure"]
    hour = departures["scheduled_datetime"].dt.floor("h")
    per_airport_hour = departures.groupby([departures["airport"], hour]).size()
    per_airline_day = departures.groupby(
        [departures["airline"], departures["scheduled_datetime"].dt.normalize()]
    ).size()

    target_hour = targets["scheduled_datetime"].dt.floor("h")
    target_day = targets["scheduled_datetime"].dt.normalize()
    airport_hour_keys = pd.MultiIndex.from_arrays([targets["airport"], target_hour])
    airline_day_keys = pd.MultiIndex.from_arrays([targets["airline"], target_day])
    return targets.assign(
        departures_same_hour=per_airport_hour.reindex(airport_hour_keys).fillna(0).to_numpy(),
        airline_departures_today=per_airline_day.reindex(airline_day_keys).fillna(0).to_numpy(),
    )


# ---------------------------------------------------------------------------- historical rates


def _trailing(
    target_keys: pd.Series,
    target_dates: pd.Series,
    history: pd.DataFrame,
    key: str,
    window_days: int,
    smoothing: int,
    stat: str,
) -> np.ndarray:
    """A smoothed statistic of `history` rows with the same key in the days before each target date.

    The window ends the day before the flight, so a row never sees its own outcome or
    any other outcome from its own day or later. Rare keys are shrunk toward the
    statistic over all keys in the same window: (n * key_stat + m * all_stat) / (n + m).
    """
    hist_dates = history["date"].to_numpy()
    values = history["value"].to_numpy(dtype=float)
    hist_keys = history[key]
    target_keys = target_keys.reset_index(drop=True)
    target_dates = target_dates.reset_index(drop=True)
    result = np.full(len(target_keys), np.nan)
    # One pass per target date: select that date's window once, then aggregate per key in it
    for day_key, rows in target_dates.groupby(target_dates).groups.items():
        day = pd.Timestamp(day_key).to_datetime64()  # type: ignore[arg-type]
        in_window = (hist_dates < day) & (hist_dates >= day - np.timedelta64(window_days, "D"))
        if not in_window.any():
            continue
        window = pd.Series(values[in_window], index=hist_keys.to_numpy()[in_window])
        prior = float(window.mean() if stat == "mean" else window.median())
        grouped = window.groupby(level=0)
        stats = grouped.mean() if stat == "mean" else grouped.median()
        counts = grouped.size()
        smoothed = (counts * stats + smoothing * prior) / (counts + smoothing)
        keys = target_keys.iloc[rows]
        result[rows] = keys.map(smoothed).fillna(prior).to_numpy(dtype=float)
    return result


def add_rate_features(targets: pd.DataFrame, context: pd.DataFrame, settings: FeatureSettings) -> pd.DataFrame:
    history = context[(context["flight_type"] == "Domestic Departure") & context["delay_minutes"].notna()].assign(
        date=lambda h: h["scheduled_datetime"].dt.normalize()
    )
    history = history.assign(route=history["airport"] + " → " + history["destination_or_origin"].fillna(""))
    target_dates = targets["scheduled_datetime"].dt.normalize()
    route = targets["airport"] + " → " + targets["destination_or_origin"].fillna("")

    delayed = history.assign(value=(history["delay_minutes"] > DELAYED_THRESHOLD_MINUTES).astype(float))
    minutes = history.assign(value=history["delay_minutes"])
    window, m = settings.rate_window_days, settings.rate_smoothing
    return targets.assign(
        airline_delay_rate_7d=_trailing(targets["airline"], target_dates, delayed, "airline", window, m, "mean"),
        route_median_delay_7d=_trailing(route, target_dates, minutes, "route", window, m, "median"),
    )


# ---------------------------------------------------------------------------- entry point


def build_features(
    targets: pd.DataFrame, context: pd.DataFrame, settings: FeatureSettings | None = None
) -> pd.DataFrame:
    """Add every feature column to `targets` (departures), using `context` (all known flights)."""
    settings = settings or FeatureSettings()
    out = targets.copy()
    for col in ("scheduled_datetime", "actual_datetime"):
        out[col] = pd.to_datetime(out[col])
    context = context.assign(
        scheduled_datetime=pd.to_datetime(context["scheduled_datetime"]),
        actual_datetime=pd.to_datetime(context["actual_datetime"]),
    )
    out = add_calendar_features(out)
    out = add_rotation_features(out, context, settings)
    out = add_congestion_features(out, context)
    out = add_rate_features(out, context, settings)
    return out
