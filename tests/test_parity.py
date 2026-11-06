"""Train/serve parity: the API's features for a flight equal the features training used for it."""

from pathlib import Path

import pandas as pd
import pytest

from aeroscrape.config import get_app_config
from aeroscrape.data.clean import clean_flights
from aeroscrape.features.build import FeatureSettings
from aeroscrape.features.preprocess import make_model_frame, variant_feature_set
from aeroscrape.serving.feature_lookup import FeatureLookup, FrameSource

RAW = Path(__file__).parent / "fixtures" / "mehrabad_4days.csv"


@pytest.fixture(scope="module")
def raw() -> pd.DataFrame:
    return pd.read_csv(RAW, dtype=str)


@pytest.fixture(scope="module")
def training_rows(raw: pd.DataFrame) -> pd.DataFrame:
    rows = make_model_frame(clean_flights(raw), get_app_config())
    # The last two days have earlier days in the fixture, so rate features are defined
    return rows[rows["scheduled_datetime"] >= "2025-06-01"].sample(50, random_state=0)


def _request(row: pd.Series, **overrides: object) -> dict:
    request = {
        col: row[col]
        for col in ("airline", "airport", "destination_or_origin", "aircraft", "flight_number", "register")
    }
    return {**request, "scheduled_datetime": str(row["scheduled_datetime"]), **overrides}


@pytest.mark.parametrize("variant", ["booking", "pre_departure"])
def test_serving_features_equal_training_features(raw, training_rows, variant: str) -> None:
    columns = variant_feature_set(get_app_config(), variant).columns
    lookup = FeatureLookup(FrameSource(raw), FeatureSettings.from_config(get_app_config()))

    served = pd.concat(
        [lookup.lookup(_request(row)).features for _, row in training_rows.iterrows()], ignore_index=True
    )

    expected = training_rows[columns].reset_index(drop=True)
    pd.testing.assert_frame_equal(served[columns], expected, check_dtype=False)
    # The check is only meaningful if the fixture exercises rotation and rate features
    assert expected["scheduled_turnaround_min"].notna().sum() >= 10
    assert expected["airline_delay_rate_7d"].notna().all()


def test_history_is_fetched_once_per_day(raw, training_rows) -> None:
    calls = []

    class CountingSource(FrameSource):
        def fetch_flights(self, start=None, end=None):
            calls.append((start, end))
            return super().fetch_flights(start, end)

    lookup = FeatureLookup(CountingSource(raw), FeatureSettings())
    same_day = training_rows[training_rows["scheduled_datetime"].dt.date == pd.Timestamp("2025-06-02").date()]
    for _, row in same_day.head(5).iterrows():
        lookup.lookup(_request(row))

    history_calls = [c for c in calls if c[1] == pd.Timestamp("2025-06-02")]
    assert len(history_calls) == 1  # four later requests reuse the cached history
    assert len(calls) == 6  # one history fetch, then one fetch of the day per request


def test_unscraped_flight_still_gets_features(raw, training_rows) -> None:
    row = training_rows.iloc[0]
    lookup = FeatureLookup(FrameSource(raw), FeatureSettings())

    result = lookup.lookup(_request(row, flight_number="NEW123", register=None))

    assert not result.matched_flight
    assert result.features.loc[0, "airline_delay_rate_7d"] == pytest.approx(row["airline_delay_rate_7d"])
    assert pd.isna(result.features.loc[0, "scheduled_turnaround_min"])  # no registration, no rotation


def test_batched_lookup_equals_training_features(raw, training_rows) -> None:
    columns = variant_feature_set(get_app_config(), "pre_departure").columns
    lookup = FeatureLookup(FrameSource(raw), FeatureSettings.from_config(get_app_config()))
    requests = [_request(row) for _, row in training_rows.iterrows()]
    requests.append(_request(training_rows.iloc[0], flight_number=None, register=None))  # unnamed flight

    result = lookup.lookup_many(requests)

    assert result.matched == [True] * len(training_rows) + [False]
    pd.testing.assert_frame_equal(
        result.features[columns].iloc[:-1].reset_index(drop=True),
        training_rows[columns].reset_index(drop=True),
        check_dtype=False,
    )
