import pandas as pd
import pytest
from fastapi.testclient import TestClient

from aeroscrape.features.preprocess import FeatureSet
from aeroscrape.serving import api

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
    features = FeatureSet(("Scheduled_Hour_of_Day", "airline_delay_rate_7d"), ("airline",))

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
    assert "airline_delay_rate_7d" in body["warnings"][0]  # not computable from one request


def test_bad_datetime_returns_400(client: TestClient) -> None:
    response = client.post("/predict", json={**PAYLOAD, "scheduled_datetime": "not a date"})
    assert response.status_code == 400


def test_predict_returns_503_without_a_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(api, "model", None)
    response = TestClient(api.app).post("/predict", json=PAYLOAD)
    assert response.status_code == 503
