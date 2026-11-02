"""Hyperparameter search scored by rolling-origin cross-validation on train+val.

The test set is never passed in, so tuning can't overfit to it. Each fold trains on
earlier flights and scores later ones, the same direction the model is used in.
"""

import json
import logging
import os
from typing import Any

import optuna
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

from aeroscrape.models.train import build_classification_pipeline, build_regression_pipeline

logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)


def _study(seed: int, direction: str) -> optuna.Study:
    return optuna.create_study(direction=direction, sampler=optuna.samplers.TPESampler(seed=seed))


def tune_regression_hyperparameters(
    X: pd.DataFrame,
    y: pd.Series,
    pipeline_config: dict[str, Any],
    n_trials: int,
    cv_splits: int,
) -> tuple[dict[str, Any], float]:
    """Return the best LightGBM parameters (defaults merged with tuned values) and their CV MAE."""
    defaults = dict(pipeline_config["lgbm_params"])
    cv = TimeSeriesSplit(n_splits=cv_splits)

    def objective(trial: optuna.Trial) -> float:
        params = defaults | {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 8, 128),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        }
        pipeline = build_regression_pipeline(pipeline_config, params)
        return -cross_val_score(pipeline, X, y, cv=cv, scoring="neg_mean_absolute_error").mean()

    study = _study(pipeline_config["random_state"], "minimize")
    study.optimize(objective, n_trials=n_trials, timeout=600)
    logger.info("Regression tuning: best CV MAE %.2f with %s", study.best_value, study.best_params)
    return defaults | study.best_params, float(study.best_value)


def tune_classification_hyperparameters(
    X: pd.DataFrame,
    y: pd.Series,
    pipeline_config: dict[str, Any],
    n_trials: int,
    cv_splits: int,
) -> tuple[dict[str, Any], float]:
    """Return the best logistic regression parameters and their CV Brier score.

    Brier score rewards well-calibrated probabilities; F1 at a fixed 0.5 threshold,
    used before, ignores them and is unstable on small folds.
    """
    defaults = dict(pipeline_config["logistic_params"])
    cv = TimeSeriesSplit(n_splits=cv_splits)

    def objective(trial: optuna.Trial) -> float:
        params = defaults | {
            "C": trial.suggest_float("C", 1e-4, 100, log=True),
            "penalty": trial.suggest_categorical("penalty", ["l1", "l2"]),
        }
        pipeline = build_classification_pipeline(pipeline_config, params)
        return -cross_val_score(pipeline, X, y, cv=cv, scoring="neg_brier_score").mean()

    study = _study(pipeline_config["random_state"], "minimize")
    study.optimize(objective, n_trials=n_trials, timeout=600)
    logger.info("Classification tuning: best CV Brier %.4f with %s", study.best_value, study.best_params)
    return defaults | study.best_params, float(study.best_value)


def save_best_params(params, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(params, f, indent=4)
    logger.info(f"Best parameters saved to {path}")
