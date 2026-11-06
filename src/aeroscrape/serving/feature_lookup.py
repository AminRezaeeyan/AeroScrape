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
        """Features for one requested domestic departure; see `lookup_many`."""
        batch = self.lookup_many([request])
        return LookupResult(batch.features, batch.matched[0], batch.context_rows)

    def lookup_many(self, requests: list[dict[str, Any]]) -> "BatchLookupResult":
        """Features for requested domestic departures, in request order.

        Each request has airport, airline, destination_or_origin, aircraft and scheduled_datetime,
        and optionally flight_number and register. If a flight was scraped, its scraped row
        (with registration) is used, exactly as training would use it. Requests are grouped by
        date, so each date's flights are fetched and featurized once for the whole batch.
        """
        targets = clean_flights(pd.DataFrame([_raw_target(r, i) for i, r in enumerate(requests)]))
        # clean_flights sorts by time; restore request order
        targets = targets.set_index(targets["flight_number"].map(_request_index)).sort_index()
        named = pd.Series([bool(r.get("flight_number")) for r in requests])
        targets["flight_number"] = [r.get("flight_number") or "REQUEST" for r in requests]
        targets.index = pd.RangeIndex(len(requests))

        features, matched, context_rows = [], pd.Series(False, index=targets.index), 0
        for _, group in targets.groupby(targets["scheduled_datetime"].dt.normalize()):
            context = self.context(group["scheduled_datetime"].min())
            if len(context):
                departures = context[context["flight_type"] == DEPARTURE].drop_duplicates(list(FLIGHT_KEY))
                key = ["airport", "flight_number", "scheduled_datetime"]
                found = group[named[group.index]].reset_index().merge(departures, on=key, suffixes=("_request", ""))
                for _, row in found.iterrows():
                    group.loc[row["index"], departures.columns] = row[departures.columns]
                    matched[row["index"]] = True
            # Training sees each departure in its own context, so requested flights are added if missing
            full = pd.concat([context, group]) if len(context) else group
            full = full.drop_duplicates(list(FLIGHT_KEY), keep="first").reset_index(drop=True)
            built = build_features(group.reset_index(drop=True), full, self.settings)
            features.append(built.set_index(group.index))
            context_rows += len(full)
        return BatchLookupResult(pd.concat(features).sort_index(), matched.tolist(), context_rows)


@dataclass
class BatchLookupResult:
    features: pd.DataFrame  # one row per request, in request order
    matched: list[bool]  # whether each requested flight was found among scraped flights
    context_rows: int


_PLACEHOLDER = "REQUEST-"


def _raw_target(request: dict[str, Any], index: int) -> dict[str, Any]:
    """A requested flight in the raw scraped format. Each gets a unique placeholder number while
    cleaning, so two requests for unnamed flights at the same time aren't merged as duplicates."""
    scheduled = pd.Timestamp(request["scheduled_datetime"])
    return {
        "scheduled_time": scheduled.strftime("%Y-%m-%d %H:%M:%S"),
        "airline": request.get("airline"),
        "flight_number": f"{_PLACEHOLDER}{index}",
        "destination_or_origin": request.get("destination_or_origin"),
        "status": None,
        "counter": None,
        "actual_time": None,
        "register": request.get("register"),
        "aircraft": request.get("aircraft"),
        "airport": request.get("airport"),
        "flight_type": DEPARTURE,
    }


def _request_index(placeholder: str) -> int:
    return int(placeholder.removeprefix(_PLACEHOLDER))


def settings_from(config: dict[str, Any]) -> FeatureSettings:
    return FeatureSettings.from_config(config)
