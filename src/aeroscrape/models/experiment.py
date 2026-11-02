"""Split, tune, train and evaluate, reusable from Airflow and from the command line.

Run locally without Airflow or MLflow:

    python -m aeroscrape.models.experiment            # tune, train, evaluate, write the report
    python -m aeroscrape.models.experiment --no-tune  # use the default parameters from config.yaml
"""

import argparse
import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.pipeline import Pipeline

from aeroscrape.config import get_app_config, resolve_path
from aeroscrape.data.clean import clean_and_engineer_features
from aeroscrape.data.split import Splits, time_split
from aeroscrape.features.preprocess import feature_columns, prepare_model_frame
from aeroscrape.models.baselines import GroupMedianRegressor, GroupRateClassifier
from aeroscrape.models.evaluate import (
    EvaluationReport,
    bootstrap_indices,
    evaluate_classification,
    evaluate_regression,
)
from aeroscrape.models.train import build_classification_pipeline, build_regression_pipeline
from aeroscrape.models.tune import tune_classification_hyperparameters, tune_regression_hyperparameters

logger = logging.getLogger(__name__)

REGRESSOR_NAME = "LightGBM"
CLASSIFIER_NAME = "Logistic regression"
# Baselines see only the columns named here, fit on the same train+val rows as the models
REGRESSION_BASELINES = {
    "Global median": [],
    "Airline × time-of-day median": ["airline", "scheduled_time_of_day"],
}
CLASSIFICATION_BASELINES = {
    "Majority class": [],
    "Airline delay rate": ["airline"],
}


@dataclass
class TunedParams:
    regression: dict[str, Any]
    classification: dict[str, Any]
    cv_scores: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {"regression": self.regression, "classification": self.classification, "cv_scores": self.cv_scores}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TunedParams":
        return cls(data["regression"], data["classification"], data.get("cv_scores", {}))


@dataclass
class ExperimentResult:
    params: TunedParams
    regressor: Pipeline
    classifier: Pipeline
    report: EvaluationReport


def make_splits(cleaned: pd.DataFrame, config: dict[str, Any]) -> Splits:
    frame = prepare_model_frame(cleaned, config["pipeline"])
    eval_cfg = config["evaluation"]
    return time_split(frame, eval_cfg["time_column"], eval_cfg["train_end"], eval_cfg["val_end"])


def _xy(df: pd.DataFrame, pipeline_config: dict[str, Any], target: str) -> tuple[pd.DataFrame, pd.Series]:
    return df[feature_columns(pipeline_config)], df[pipeline_config[target]]


def tune_models(splits: Splits, config: dict[str, Any], enabled: bool | None = None) -> TunedParams:
    """Tune on train+val with rolling-origin CV. The test split is not used."""
    pipeline_cfg = config["pipeline"]
    tuning = pipeline_cfg["tuning"]
    if not (tuning["enabled"] if enabled is None else enabled):
        logger.info("Tuning disabled; using default parameters from config.yaml")
        return TunedParams(dict(pipeline_cfg["lgbm_params"]), dict(pipeline_cfg["logistic_params"]), {})

    cv_splits = config["evaluation"]["cv_splits"]
    X, y_reg = _xy(splits.train_val, pipeline_cfg, "target_regression")
    _, y_cls = _xy(splits.train_val, pipeline_cfg, "target_classification")
    reg_params, reg_cv = tune_regression_hyperparameters(
        X, y_reg, pipeline_cfg, tuning["n_trials_regression"], cv_splits
    )
    cls_params, cls_cv = tune_classification_hyperparameters(
        X, y_cls, pipeline_cfg, tuning["n_trials_classification"], cv_splits
    )
    return TunedParams(reg_params, cls_params, {"cv_mae": reg_cv, "cv_brier": cls_cv})


def train_and_evaluate(splits: Splits, params: TunedParams, config: dict[str, Any]) -> ExperimentResult:
    """Fit models and baselines on train+val, then score everything once on test."""
    pipeline_cfg, eval_cfg = config["pipeline"], config["evaluation"]
    X_fit, y_reg_fit = _xy(splits.train_val, pipeline_cfg, "target_regression")
    _, y_cls_fit = _xy(splits.train_val, pipeline_cfg, "target_classification")
    X_test, y_reg_test = _xy(splits.test, pipeline_cfg, "target_regression")
    _, y_cls_test = _xy(splits.test, pipeline_cfg, "target_classification")

    regressor = build_regression_pipeline(pipeline_cfg, params.regression).fit(X_fit, y_reg_fit)
    classifier = build_classification_pipeline(pipeline_cfg, params.classification).fit(X_fit, y_cls_fit)

    reg_predictions = {
        name: GroupMedianRegressor(columns).fit(X_fit, y_reg_fit).predict(X_test)
        for name, columns in REGRESSION_BASELINES.items()
    }
    reg_predictions[REGRESSOR_NAME] = regressor.predict(X_test)

    cls_predictions = {}
    for name, columns in CLASSIFICATION_BASELINES.items():
        baseline = GroupRateClassifier(columns).fit(X_fit, y_cls_fit)
        cls_predictions[name] = (baseline.predict(X_test), baseline.predict_proba(X_test)[:, 1])
    cls_predictions[CLASSIFIER_NAME] = (classifier.predict(X_test), classifier.predict_proba(X_test)[:, 1])

    indices = bootstrap_indices(len(splits.test), eval_cfg["bootstrap_samples"], pipeline_cfg["random_state"])
    confidence = eval_cfg["confidence_level"]
    report = EvaluationReport(
        split_summary=splits.summary(eval_cfg["time_column"]),
        tasks=[
            evaluate_regression(
                y_reg_test.to_numpy(), reg_predictions, list(REGRESSION_BASELINES), indices, confidence
            ),
            evaluate_classification(
                y_cls_test.to_numpy(),
                cls_predictions,
                list(CLASSIFICATION_BASELINES),
                indices,
                confidence,
                pipeline_cfg["delay_threshold_minutes"],
            ),
        ],
        bootstrap_samples=eval_cfg["bootstrap_samples"],
        confidence=confidence,
        extra_sections=_tuning_section(params),
    )
    return ExperimentResult(params, regressor, classifier, report)


def _tuning_section(params: TunedParams) -> list[str]:
    if not params.cv_scores:
        return ["## Tuning", "", "Tuning was disabled; default parameters from `config.yaml` were used."]
    return [
        "## Tuning",
        "",
        "Parameters were chosen by rolling-origin cross-validation on train+val (test unused): "
        f"CV MAE {params.cv_scores['cv_mae']:.1f} min, CV Brier {params.cv_scores['cv_brier']:.3f}.",
    ]


def run_experiment(cleaned: pd.DataFrame, config: dict[str, Any], tune: bool | None = None) -> ExperimentResult:
    splits = make_splits(cleaned, config)
    params = tune_models(splits, config, enabled=tune)
    return train_and_evaluate(splits, params, config)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    config = get_app_config()
    parser = argparse.ArgumentParser(description="Train and evaluate the delay models on a time split.")
    parser.add_argument("--raw", default=config["data_paths"]["raw_data"], help="Raw scraped CSV")
    parser.add_argument("--output", default=config["evaluation"]["report_path"], help="Markdown report path")
    tuning = parser.add_mutually_exclusive_group()
    tuning.add_argument("--tune", dest="tune", action="store_true", default=None, help="Force tuning on")
    tuning.add_argument("--no-tune", dest="tune", action="store_false", help="Use default parameters")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        cleaned_path = Path(tmp) / "cleaned.csv"
        clean_and_engineer_features(resolve_path(args.raw), cleaned_path)
        cleaned = pd.read_csv(cleaned_path)

    result = run_experiment(cleaned, config, tune=args.tune)
    output = resolve_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(result.report.to_markdown(), encoding="utf-8")
    print(result.report.to_markdown())
    logger.info("Report written to %s", output)


if __name__ == "__main__":
    main()
