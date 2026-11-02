"""Promotion and rollback against a real (SQLite-backed) MLflow registry."""

import mlflow
import pytest
from mlflow import MlflowClient
from mlflow.pyfunc import PythonModel

from aeroscrape.models.registry import CHAMPION, alias_version, promote, status


class Constant(PythonModel):
    def predict(self, context, model_input, params=None):
        return model_input


@pytest.fixture
def registry(tmp_path):
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    client = MlflowClient()
    for _ in range(3):
        with mlflow.start_run():
            mlflow.pyfunc.log_model(
                name="m", python_model=Constant(), registered_model_name="delay", pip_requirements=[]
            )
    return client


def test_promote_moves_the_alias_and_records_the_previous_version(registry: MlflowClient) -> None:
    promote(registry, "delay", "1", reason="first")
    promote(registry, "delay", "3", reason="passed gate")

    assert alias_version(registry, "delay", CHAMPION) == "3"
    tags = registry.get_model_version("delay", "3").tags
    assert tags["replaced_version"] == "1" and tags["promoted_reason"] == "passed gate"


def test_rollback_is_a_promotion_of_an_older_version(registry: MlflowClient) -> None:
    promote(registry, "delay", "3", reason="passed gate")
    promote(registry, "delay", "1", reason="manual rollback")

    assert alias_version(registry, "delay", CHAMPION) == "1"
    assert "production  version 1" in status(registry, "delay")


def test_missing_alias_is_none(registry: MlflowClient) -> None:
    assert alias_version(registry, "delay", CHAMPION) is None
