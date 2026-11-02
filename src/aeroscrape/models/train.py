import json
import logging
import os
from typing import Any

import joblib
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from aeroscrape.config import get_app_config, resolve_path
from aeroscrape.features.preprocess import build_preprocessor

logger = logging.getLogger(__name__)


def build_regression_pipeline(pipeline_config: dict[str, Any], params: dict[str, Any]) -> Pipeline:
    """Preprocessor + LightGBM regressor. Fitting the whole pipeline keeps the preprocessor
    from ever seeing the rows a model is scored on, including inside CV folds."""
    model = lgb.LGBMRegressor(**params, random_state=pipeline_config["random_state"])
    return Pipeline([("preprocess", build_preprocessor(pipeline_config)), ("model", model)])


def build_classification_pipeline(pipeline_config: dict[str, Any], params: dict[str, Any]) -> Pipeline:
    model = LogisticRegression(**params, random_state=pipeline_config["random_state"])
    return Pipeline([("preprocess", build_preprocessor(pipeline_config)), ("model", model)])


def save_model(model, model_name, subfolder=""):
    app_config = get_app_config()
    base_model_output_dir = app_config["pipeline"]["model_output_dir"]
    model_dir = os.path.join(resolve_path(base_model_output_dir), subfolder)
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, f"{model_name}.joblib")
    joblib.dump(model, model_path)
    logger.info(f"Model '{model_name}' saved to {model_path}")
    return model_path


def save_metrics(metrics, model_name, subfolder=""):
    app_config = get_app_config()
    base_model_output_dir = app_config["pipeline"]["model_output_dir"]

    metrics_dir = os.path.join(resolve_path(base_model_output_dir), subfolder)
    os.makedirs(metrics_dir, exist_ok=True)

    metrics_path = os.path.join(metrics_dir, f"{model_name}_metrics.json")

    serializable_metrics = {k: (float(v) if hasattr(v, "item") else v) for k, v in metrics.items()}

    with open(metrics_path, "w") as f:
        json.dump(serializable_metrics, f, indent=4)
    logger.info(f"Metrics for '{model_name}' saved to {metrics_path}")
    return metrics_path
