"""The drift reference: a sample of a model's training inputs, kept with its MLflow run.

It is rebuilt from the configured data sources with the training code (the model's
training window is in its registry tags) the first time a version is monitored, then
stored as an artifact of the version's run, so later checks just download it.
"""

import logging
import tempfile
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
from mlflow import MlflowClient

from aeroscrape.data.sources import load_cleaned_flights
from aeroscrape.features.preprocess import make_model_frame, prepare_input

logger = logging.getLogger(__name__)

ARTIFACT = "monitoring/reference_inputs.parquet"


def build_reference(config: dict[str, Any], model: Any, trained_from: str, trained_to: str, rows: int) -> pd.DataFrame:
    """Model inputs of up to `rows` training flights scheduled in [trained_from, trained_to]."""
    bundle = model.unwrap_python_model().bundle
    flights, _ = load_cleaned_flights(config)
    frame = make_model_frame(flights, config)
    time_column = config["evaluation"]["time_column"]
    window = frame[frame[time_column].between(pd.Timestamp(trained_from), pd.Timestamp(trained_to))]
    if window.empty:
        raise ValueError(f"No training rows between {trained_from} and {trained_to} in the configured sources")
    sample = window.sample(min(rows, len(window)), random_state=config["pipeline"]["random_state"])
    return prepare_input(sample, bundle.features).reset_index(drop=True)


def reference_inputs(config: dict[str, Any], client: MlflowClient, name: str, version: str) -> pd.DataFrame:
    """The reference for a registered model version, built and stored on first use."""
    mv = client.get_model_version(name, version)
    try:
        path = mlflow.artifacts.download_artifacts(run_id=mv.run_id, artifact_path=ARTIFACT)
        return pd.read_parquet(path)
    except Exception:
        logger.info("No drift reference stored for version %s yet; building it from its training window", version)

    model = mlflow.pyfunc.load_model(f"models:/{name}/{version}")
    reference = build_reference(
        config,
        model,
        mv.tags["training_from"],
        mv.tags["training_to"],
        config["monitoring"]["drift"]["reference_rows"],
    )
    with tempfile.TemporaryDirectory() as tmp:
        local = Path(tmp) / Path(ARTIFACT).name
        reference.to_parquet(local, index=False)
        client.log_artifact(mv.run_id, str(local), artifact_path=str(Path(ARTIFACT).parent))
    return reference
