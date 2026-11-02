"""Model inputs: target columns and the feature preprocessor."""

import logging
from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)


def load_data(data_path):
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    logger.info(f"Data loaded successfully. Shape: {df.shape}")
    return df


def prepare_model_frame(df: pd.DataFrame, pipeline_config: dict[str, Any]) -> pd.DataFrame:
    """Add the classification target and derived inputs that the cleaner doesn't produce."""
    out = df.copy()
    scheduled = pd.to_datetime(out["scheduled_datetime"])
    out[pipeline_config["target_classification"]] = (
        out[pipeline_config["target_regression"]] > pipeline_config["delay_threshold_minutes"]
    ).astype(int)
    # Previously parsed from scheduled_time with an HH:MM:SS format that never matched,
    # so the feature was 0 for every training row while the API used the real minute.
    out["scheduled_minute"] = scheduled.dt.minute
    return out


def feature_columns(pipeline_config: dict[str, Any]) -> list[str]:
    return list(pipeline_config["numerical_features"]) + list(pipeline_config["categorical_features"])


def build_preprocessor(pipeline_config: dict[str, Any]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), list(pipeline_config["numerical_features"])),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                list(pipeline_config["categorical_features"]),
            ),
        ],
        remainder="drop",
    )
