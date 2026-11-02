from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from aeroscrape.features.build import FeatureSettings
from aeroscrape.features.preprocess import FeatureSet
from aeroscrape.serving import api
from aeroscrape.serving.feature_lookup import FeatureLookup, FrameSource

PAYLOAD = {
    "airline": "ماهان",
    "airport": "فرودگاه مهرآباد",
    "destination_or_origin": "مشهد",
    "aircraft": "RJ85",
    "scheduled_datetime": "2025-06-01 08:00:00",
}


class FakeBundle:
    variant = "booking"
    threshold = 0.6
    features = FeatureSet(("Scheduled_Hour_of_Day", "airline_delay_rate_7d", "scheduled_turnaround_min"), ("airline",))

    def prepare_input(self, X: pd.DataFrame) -> pd.DataFrame:
        return X

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {"p_delay_over_15": [0.7], "likely_delayed": [True], "p10": [5.0], "p50": [25.0], "p90": [70.0]}
        )

    def explain(self, X: pd.DataFrame, top: int = 8) -> list:
        return [[{"feature": "airline", "value": "ماهان", "contribution": 0.4}]]


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    # Not used as a context manager, so startup never contacts MLflow
    fake = FakeBundle()
    monkeypatch.setattr(api, "model", fake)
    monkeypatch.setattr(api, "bundle", fake)
    monkeypatch.setattr(api, "model_version", "7")
    return TestClient(api.app)


def test_prediction_has_probability_range_and_version(client: TestClient) -> None:
    body = client.post("/predict", json=PAYLOAD).json()

    assert body["model_version"] == "7"
    assert body["p_delay_over_15"] == 0.7 and body["likely_delayed"] is True
    assert body["delay_minutes"] == {"p10": 5.0, "p50": 25.0, "p90": 70.0}
    assert body["top_factors"][0]["feature"] == "airline"
    assert "Flight database unavailable" in body["warnings"][0]  # no lookup in this test


def test_bad_datetime_returns_400(client: TestClient) -> None:
    response = client.post("/predict", json={**PAYLOAD, "scheduled_datetime": "not a date"})
    assert response.status_code == 400


def test_predict_returns_503_without_a_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(api, "model", None)
    response = TestClient(api.app).post("/predict", json=PAYLOAD)
    assert response.status_code == 503


def test_features_come_from_the_flight_database(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    raw = pd.read_csv(Path(__file__).parent / "fixtures" / "mehrabad_4days.csv", dtype=str)
    monkeypatch.setattr(api, "lookup", FeatureLookup(FrameSource(raw), FeatureSettings()))
    seen = {}

    def capture(X: pd.DataFrame) -> pd.DataFrame:
        seen["X"] = X
        return FakeBundle.predict(None, X)

    monkeypatch.setattr(api.bundle, "predict", capture)
    flight = raw[(raw["flight_type"] == "Domestic Departure") & raw["register"].notna()].iloc[-1]
    response = client.post(
        "/predict",
        json={
            "airline": flight["airline"],
            "airport": flight["airport"],
            "destination_or_origin": flight["destination_or_origin"],
            "aircraft": flight["aircraft"],
            "scheduled_datetime": flight["scheduled_time"],
            "flight_number": flight["flight_number"],
        },
    )

    assert response.status_code == 200
    assert not any("unavailable" in w or "not found" in w for w in response.json()["warnings"])
    assert seen["X"]["airline_delay_rate_7d"].notna().all()


def test_unknown_flight_is_reported(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    raw = pd.read_csv(Path(__file__).parent / "fixtures" / "mehrabad_4days.csv", dtype=str)
    monkeypatch.setattr(api, "lookup", FeatureLookup(FrameSource(raw), FeatureSettings()))
    body = client.post("/predict", json={**PAYLOAD, "flight_number": "NOPE1"}).json()
    assert any("not found" in w for w in body["warnings"])
