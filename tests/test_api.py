import json
import logging
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from aeroscrape.features.build import FeatureSettings
from aeroscrape.features.preprocess import FeatureSet
from aeroscrape.serving import api
from aeroscrape.serving.feature_lookup import FeatureLookup, FrameSource
from aeroscrape.serving.logs import JsonFormatter

FLIGHT = {
    "airline": "ماهان",
    "airport": "فرودگاه مهرآباد",
    "destination_or_origin": "مشهد",
    "aircraft": "RJ85",
    "scheduled_datetime": "2025-06-01 08:00:00",
}
MEHRABAD = Path(__file__).parent / "fixtures" / "mehrabad_4days.csv"


class FakeBundle:
    """Stands in for both the pyfunc model and the unwrapped bundle."""

    variant = "booking"
    threshold = 0.6
    features = FeatureSet(("Scheduled_Hour_of_Day", "airline_delay_rate_7d"), ("airline", "airport"))

    def prepare_input(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.reindex(columns=self.features.columns).astype({"airline": str, "airport": str})

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        n = len(X)
        return pd.DataFrame(
            {
                "p_delay_over_15": [0.7] * n,
                "likely_delayed": [True] * n,
                "p10": [5.0] * n,
                "p50": [25.0] * n,
                "p90": [70.0] * n,
            }
        )

    def explain(self, X: pd.DataFrame, top: int = 8) -> list:
        return [[{"feature": "airline", "value": "ماهان", "contribution": 0.4}] for _ in range(len(X))]


KNOWN = {"airline": {"ماهان", "آتا"}, "airport": {"فرودگاه مهرآباد"}}


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    # Not used as a context manager, so startup never contacts MLflow or Postgres
    fake = FakeBundle()
    monkeypatch.setattr(api, "served", api.ServedModel(fake, fake, "7", KNOWN))
    monkeypatch.setattr(api, "lookup", None)
    monkeypatch.setattr(api, "API_KEY", None)
    return TestClient(api.app)


def test_prediction_has_probability_range_and_version(client: TestClient) -> None:
    body = client.post("/v1/predict", json=FLIGHT).json()

    assert body["model_version"] == "7" and body["decision_threshold"] == 0.6
    assert body["p_delay_over_15"] == 0.7 and body["likely_delayed"] is True
    assert body["delay_minutes"] == {"p10": 5.0, "p50": 25.0, "p90": 70.0}
    assert body["top_factors"] is None and body["explanation_units"] is None  # SHAP is opt-in
    assert "Flight database unavailable" in body["warnings"][0]


def test_explanations_on_request(client: TestClient) -> None:
    body = client.post("/v1/predict?explain=true", json=FLIGHT).json()
    assert body["top_factors"][0]["feature"] == "airline"
    assert "log-odds" in body["explanation_units"]


@pytest.mark.parametrize(
    "change",
    [
        {"scheduled_datetime": "not a date"},
        {"airline": "   "},
        {"unexpected_field": 1},
        {"flight_number": "X" * 50},
    ],
)
def test_invalid_input_is_rejected_with_422(client: TestClient, change: dict) -> None:
    assert client.post("/v1/predict", json=FLIGHT | change).status_code == 422


def test_unseen_categories_are_reported(client: TestClient) -> None:
    body = client.post("/v1/predict", json=FLIGHT | {"airline": "New Air"}).json()
    assert "unseen airline: New Air" in body["warnings"]


def test_batch_scores_every_flight(client: TestClient) -> None:
    flights = [FLIGHT, FLIGHT | {"airline": "آتا"}, FLIGHT | {"scheduled_datetime": "2025-06-02T23:30:00"}]
    body = client.post("/v1/predict/batch", json={"flights": flights}).json()

    assert body["model_version"] == "7"
    assert len(body["predictions"]) == 3


def test_batch_size_is_limited(client: TestClient) -> None:
    response = client.post("/v1/predict/batch", json={"flights": [FLIGHT] * (api.MAX_BATCH + 1)})
    assert response.status_code == 422


def test_predict_returns_503_without_a_model(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(api, "served", None)
    assert client.post("/v1/predict", json=FLIGHT).status_code == 503


def test_api_key_is_required_when_configured(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(api, "API_KEY", "s3cret")

    assert client.post("/v1/predict", json=FLIGHT).status_code == 401
    assert client.post("/v1/predict", json=FLIGHT, headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.post("/v1/predict", json=FLIGHT, headers={"X-API-Key": "s3cret"}).status_code == 200
    assert client.get("/health").status_code == 200  # probes stay open


def test_cors_allows_only_the_demo_origin(client: TestClient) -> None:
    headers = {"Access-Control-Request-Method": "POST"}
    allowed = client.options("/v1/predict", headers=headers | {"Origin": "http://localhost:8501"})
    other = client.options("/v1/predict", headers=headers | {"Origin": "http://evil.example"})
    assert allowed.headers.get("access-control-allow-origin") == "http://localhost:8501"
    assert "access-control-allow-origin" not in other.headers


def test_request_id_is_echoed_and_logged_as_json(client: TestClient, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="aeroscrape.serving.api"):
        response = client.post("/v1/predict", json=FLIGHT, headers={"X-Request-ID": "abc123"})

    assert response.headers["X-Request-ID"] == "abc123"
    record = next(r for r in caplog.records if r.getMessage() == "request")
    line = json.loads(JsonFormatter().format(record))
    assert line["path"] == "/v1/predict" and line["status"] == 200
    assert line["model_version"] == "7" and "latency_ms" in line and "mean_p_delay" in line


def test_health_and_readiness(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    assert client.get("/health").json() == {"status": "alive"}
    assert client.get("/ready").status_code == 503  # no feature lookup in this test

    monkeypatch.setattr(api, "lookup", FeatureLookup(FrameSource(pd.read_csv(MEHRABAD, dtype=str)), FeatureSettings()))
    body = client.get("/ready").json()
    assert body["status"] == "ready" and body["model_version"] == "7"


def test_features_come_from_the_flight_database(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    raw = pd.read_csv(MEHRABAD, dtype=str)
    monkeypatch.setattr(api, "lookup", FeatureLookup(FrameSource(raw), FeatureSettings()))
    seen = {}

    def capture(X: pd.DataFrame) -> pd.DataFrame:
        seen["X"] = X
        return FakeBundle().predict(X)

    monkeypatch.setattr(api.served.model, "predict", capture)
    flight = raw[(raw["flight_type"] == "Domestic Departure") & raw["register"].notna()].iloc[-1]
    response = client.post(
        "/v1/predict",
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
    raw = pd.read_csv(MEHRABAD, dtype=str)
    monkeypatch.setattr(api, "lookup", FeatureLookup(FrameSource(raw), FeatureSettings()))
    body = client.post("/v1/predict", json=FLIGHT | {"flight_number": "NOPE1"}).json()
    assert any("not found" in w for w in body["warnings"])


def test_promoted_version_is_swapped_in(monkeypatch: pytest.MonkeyPatch) -> None:
    class Registry:
        version = "7"

        def get_model_version_by_alias(self, name, alias):
            return type("V", (), {"version": self.version})()

    class Loaded:
        def unwrap_python_model(self):
            return type(
                "W", (), {"bundle": type("B", (), {"variant": "booking", "known_categories": lambda self: {}})()}
            )()

    registry = Registry()
    monkeypatch.setattr(api, "_registry", lambda: (registry, "flight-delay-model", "production"))
    monkeypatch.setattr(api.mlflow.pyfunc, "load_model", lambda uri: Loaded())
    monkeypatch.setattr(api, "served", None)

    assert api.refresh_if_promoted() and api.served.version == "7"
    assert not api.refresh_if_promoted()  # unchanged alias: no reload
    registry.version = "8"
    assert api.refresh_if_promoted() and api.served.version == "8"


def test_failed_registry_check_keeps_the_current_model(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeBundle()
    current = api.ServedModel(fake, fake, "7", KNOWN)
    monkeypatch.setattr(api, "served", current)

    def broken():
        raise ConnectionError("MLflow down")

    monkeypatch.setattr(api, "_registry", broken)
    assert not api.refresh_if_promoted()
    assert api.served is current
