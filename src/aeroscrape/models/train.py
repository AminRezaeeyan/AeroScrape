import json
import logging
import os
import warnings
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from aeroscrape.config import get_app_config, resolve_path
from aeroscrape.features.preprocess import FeatureSet, build_preprocessor

logger = logging.getLogger(__name__)

# LightGBM 4.6 invents feature names (Column_0, ...) even when fitted on an array, so scikit-learn
# 1.6 warns on every predict that "X does not have valid feature names". Our models always get the
# same ColumnTransformer array at fit and predict time, so the warning is a false alarm; only this
# exact message is silenced, so a real fit/predict mismatch elsewhere still warns.
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but LGBM(Regressor|Classifier) was fitted with feature names",
    category=UserWarning,
)


def build_regression_pipeline(
    pipeline_config: dict[str, Any], params: dict[str, Any], features: FeatureSet
) -> Pipeline:
    """Preprocessor + LightGBM regressor. Fitting the whole pipeline keeps the preprocessor
    from ever seeing the rows a model is scored on, including inside CV folds."""
    model = lgb.LGBMRegressor(**params, random_state=pipeline_config["random_state"])
    return Pipeline([("preprocess", build_preprocessor(features)), ("model", model)])


# Delays are skewed (median ~20 min, max 537). Quantiles survive a monotone transform, but the
# trees may fit the log scale better; the experiment picks the transform with the lower validation MAE.
LOG_OFFSET = 61.0  # the cleaner keeps delays >= -60, so delay + 61 > 0
CLIP_MINUTES = 300.0


def log_delay(y):
    return np.log(np.asarray(y, dtype=float) + LOG_OFFSET)


def exp_delay(z):
    return np.exp(np.asarray(z, dtype=float)) - LOG_OFFSET


def clip_delay(y):
    return np.minimum(np.asarray(y, dtype=float), CLIP_MINUTES)


def identity(y):
    return np.asarray(y, dtype=float)


TARGET_TRANSFORMS = {
    "none": None,
    "log": (log_delay, exp_delay),
    "clip_300": (clip_delay, identity),
}


def build_quantile_regressor(
    pipeline_config: dict[str, Any],
    params: dict[str, Any],
    features: FeatureSet,
    quantile: float,
    transform: str = "none",
):
    """A LightGBM model of one delay quantile, optionally on a transformed target."""
    quantile_params = params | {"objective": "quantile", "alpha": quantile, "metric": "quantile"}
    pipeline = build_regression_pipeline(pipeline_config, quantile_params, features)
    functions = TARGET_TRANSFORMS[transform]
    if functions is None:
        return pipeline
    func, inverse = functions
    return TransformedTargetRegressor(regressor=pipeline, func=func, inverse_func=inverse, check_inverse=False)


def build_lgbm_classification_pipeline(
    pipeline_config: dict[str, Any], params: dict[str, Any], features: FeatureSet
) -> Pipeline:
    model = lgb.LGBMClassifier(**params, random_state=pipeline_config["random_state"])
    return Pipeline([("preprocess", build_preprocessor(features)), ("model", model)])


def build_classification_pipeline(
    pipeline_config: dict[str, Any], params: dict[str, Any], features: FeatureSet
) -> Pipeline:
    model = LogisticRegression(**params, random_state=pipeline_config["random_state"])
    return Pipeline([("preprocess", build_preprocessor(features)), ("model", model)])


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
