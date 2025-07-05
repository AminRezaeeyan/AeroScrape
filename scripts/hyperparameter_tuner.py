import optuna
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error, f1_score
import logging
import json
import os

logger = logging.getLogger(__name__)

def tune_regression_hyperparameters(X_train, y_train, X_val, y_val, n_trials):
    def objective(trial):
        params = {
            "objective": "regression_l1",
            "metric": "mae",
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 20, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "verbose": -1,
            "n_jobs": -1,
        }
        model = lgb.LGBMRegressor(**params)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        mae = mean_absolute_error(y_val, preds)
        return mae

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, timeout=600)
    logger.info(f"Regression tuning complete. Best MAE: {study.best_value}")
    logger.info(f"Best params: {study.best_params}")
    return study.best_params

def tune_classification_hyperparameters(X_train, y_train, X_val, y_val, n_trials):
    def objective(trial):
        params = {
            "solver": "liblinear",
            "C": trial.suggest_float("C", 1e-5, 100, log=True),
            "penalty": trial.suggest_categorical("penalty", ["l1", "l2"]),
        }
        model = LogisticRegression(**params, max_iter=200)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)

        f1 = f1_score(y_val, preds, zero_division=0)
        return f1

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, timeout=600)
    logger.info(f"Classification tuning complete. Best F1-Score: {study.best_value}")
    logger.info(f"Best params: {study.best_params}")
    return study.best_params

def save_best_params(params, path):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(params, f, indent=4)
        logger.info(f"Best parameters saved to {path}")
    except Exception as e:
        logger.error(f"Failed to save parameters to {path}: {e}")
        raise