import pytest
from fastapi.testclient import TestClient

from aeroscrape.config import get_app_config
from aeroscrape.serving import api

PAYLOAD = {
    "airline": "ماهان",
    "airport": "فرودگاه مهرآباد",
    "destination_or_origin": "مشهد",
    "aircraft": "RJ85",
    "scheduled_datetime": "2025-06-01 08:00:00",
}


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    # Not used as a context manager, so startup never contacts MLflow
    monkeypatch.setattr(api, "model", object())
    monkeypatch.setattr(api, "preprocessor", object())
    monkeypatch.setattr(api, "app_config", get_app_config())
    return TestClient(api.app)


def test_bad_datetime_returns_400(client: TestClient) -> None:
    response = client.post("/predict", json={**PAYLOAD, "scheduled_datetime": "not a date"})
    assert response.status_code == 400


def test_predict_returns_503_without_a_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(api, "model", None)
    response = TestClient(api.app).post("/predict", json=PAYLOAD)
    assert response.status_code == 503
