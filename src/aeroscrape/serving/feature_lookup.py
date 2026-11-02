"""Build serving features with the same code as training.

Training calls `build_features(departures, all_flights)`. Serving does the same for one
requested departure: it fetches the flights the features look at from Postgres, cleans
them with the training cleaner, and calls `build_features` with them as context. There is
no second implementation of any feature, so train/serve skew can only come from the data,
and `tests/test_parity.py` checks that it doesn't.

Which flights a departure's features look at:

- rates: domestic departures from the `rate_window_days` days before its date;
- rotation and live delays: the same aircraft's flights in the hours before it, and its
  other departures that day.

Earlier days change rarely, so they are cached per date for `history_ttl_seconds`; the
flight's own day (and the hours before midnight) is fetched on every request, so live
delays are current. This is a deliberately small feature store: at about 600 flights a
day a week of history is a few thousand rows. A precomputed snapshot table refreshed by
Airflow is the next step if request latency or volume grows.
"""

import logging
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol

import pandas as pd

from aeroscrape.data.clean import FLIGHT_KEY, clean_flights
from aeroscrape.features.build import FeatureSettings, build_features

logger = logging.getLogger(__name__)

DEPARTURE = "Domestic Departure"


class FlightSource(Protocol):
    def fetch_flights(self, start: datetime | None = None, end: datetime | None = None) -> pd.DataFrame:
        """Raw scraped flights scheduled in [start, end), in the scraper's CSV format."""
        ...


class FrameSource:
    """A FlightSource over an in-memory raw DataFrame (tests, backfills, parity checks)."""

    def __init__(self, raw: pd.DataFrame):
        self.raw = raw
        self.scheduled = pd.to_datetime(raw["scheduled_time"])

    def fetch_flights(self, start: datetime | None = None, end: datetime | None = None) -> pd.DataFrame:
        mask = pd.Series(True, index=self.raw.index)
        if start is not None:
            mask &= self.scheduled >= pd.Timestamp(start)
        if end is not None:
            mask &= self.scheduled < pd.Timestamp(end)
        return self.raw[mask]


@dataclass
class LookupResult:
    features: pd.DataFrame  # one row: the target departure with every feature column
    matched_flight: bool  # the requested flight was found among scraped flights
    context_rows: int


class FeatureLookup:
    def __init__(self, source: FlightSource, settings: FeatureSettings, history_ttl_seconds: float = 900):
        self.source = source
        self.settings = settings
        self.history_ttl_seconds = history_ttl_seconds
        self._history: dict[pd.Timestamp, tuple[float, pd.DataFrame]] = {}

    def _history_for(self, day: pd.Timestamp) -> pd.DataFrame:
        """Cleaned flights from the days before `day` that rate features use, cached per day."""
        cached = self._history.get(day)
        if cached and time.monotonic() - cached[0] < self.history_ttl_seconds:
            return cached[1]
        raw = self.source.fetch_flights(day - pd.Timedelta(days=self.settings.rate_window_days), day)
        cleaned = clean_flights(raw) if len(raw) else pd.DataFrame()
        self._history = {d: v for d, v in self._history.items() if d >= day - pd.Timedelta(days=1)}
        self._history[day] = (time.monotonic(), cleaned)
        return cleaned

    def context(self, scheduled: pd.Timestamp) -> pd.DataFrame:
        """Every cleaned flight the features of a departure at `scheduled` can look at."""
        day = scheduled.normalize()
        lookback = pd.Timedelta(
            hours=max(self.settings.inbound_window_hours, self.settings.previous_departure_window_hours)
        )
        recent_raw = self.source.fetch_flights(min(day, scheduled - lookback), day + pd.Timedelta(days=1))
        parts = [self._history_for(day)]
        if len(recent_raw):
            parts.append(clean_flights(recent_raw))
        parts = [p for p in parts if len(p)]
        if not parts:
            return pd.DataFrame()
        # The day before is in both parts; the fresh copy wins
        return pd.concat(parts[::-1]).drop_duplicates(list(FLIGHT_KEY), keep="first").reset_index(drop=True)

    def lookup(self, request: dict[str, Any]) -> LookupResult:
        """Features for a requested domestic departure.

        `request` has airport, airline, destination_or_origin, aircraft and scheduled_datetime,
        and optionally flight_number and register. If the flight was scraped, its scraped row
        (with registration) is used, exactly as training would use it.
        """
        scheduled = pd.Timestamp(request["scheduled_datetime"])
        context = self.context(scheduled)
        target_raw = pd.DataFrame(
            [
                {
                    "scheduled_time": scheduled.strftime("%Y-%m-%d %H:%M:%S"),
                    "airline": request.get("airline"),
                    "flight_number": request.get("flight_number") or "REQUEST",
                    "destination_or_origin": request.get("destination_or_origin"),
                    "status": None,
                    "counter": None,
                    "actual_time": None,
                    "register": request.get("register"),
                    "aircraft": request.get("aircraft"),
                    "airport": request.get("airport"),
                    "flight_type": DEPARTURE,
                }
            ]
        )
        target = clean_flights(target_raw)

        matched = False
        if len(context) and request.get("flight_number"):
            key = target.iloc[0]
            same = context[
                (context["flight_type"] == DEPARTURE)
                & (context["airport"] == key["airport"])
                & (context["flight_number"] == key["flight_number"])
                & (context["scheduled_datetime"] == key["scheduled_datetime"])
            ]
            if len(same):
                target, matched = same.head(1).reset_index(drop=True), True

        # Training sees each departure in its own context, so the request's flight is added if missing
        full = pd.concat([context, target]) if len(context) else target
        full = full.drop_duplicates(list(FLIGHT_KEY), keep="first").reset_index(drop=True)
        features = build_features(target, full, self.settings)
        return LookupResult(features, matched, len(full))


def settings_from(config: dict[str, Any]) -> FeatureSettings:
    return FeatureSettings.from_config(config)
