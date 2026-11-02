"""Split, tune, train and evaluate, reusable from Airflow and from the command line.

Run locally without Airflow or MLflow:

    python -m aeroscrape.models.experiment            # tune, train, evaluate, write the report
    python -m aeroscrape.models.experiment --no-tune  # use the default parameters from config.yaml
"""

import argparse
import logging
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from aeroscrape.config import get_app_config, resolve_path
from aeroscrape.data.clean import CleaningAudit, clean_flights_file, read_flights
from aeroscrape.data.split import Splits, time_split
from aeroscrape.features.preprocess import (
    FeatureSet,
    ablation_steps,
    feature_coverage,
    make_model_frame,
    variant_feature_set,
)
from aeroscrape.models.baselines import GroupMedianRegressor, GroupRateClassifier
from aeroscrape.models.evaluate import (
    EvaluationReport,
    ablation_markdown,
    bootstrap_indices,
    evaluate_classification,
    evaluate_regression,
)
from aeroscrape.models.train import build_classification_pipeline, build_regression_pipeline
from aeroscrape.models.tune import tune_classification_hyperparameters, tune_regression_hyperparameters

logger = logging.getLogger(__name__)

VARIANT_LABELS = {"booking": "at booking", "pre_departure": "1 h before departure"}
# Baselines see only the columns named here, fit on the same train+val rows as the models
REGRESSION_BASELINES = {
    "Global median": [],
    "Airline × time-of-day median": ["airline", "scheduled_time_of_day"],
}
CLASSIFICATION_BASELINES = {
    "Majority class": [],
    "Airline delay rate": ["airline"],
}


def regressor_name(variant: str) -> str:
    return f"LightGBM ({VARIANT_LABELS.get(variant, variant)})"


def classifier_name(variant: str) -> str:
    return f"Logistic regression ({VARIANT_LABELS.get(variant, variant)})"


@dataclass
class TunedParams:
    """Per variant: {"regression": {...}, "classification": {...}}, plus CV scores when tuned."""

    by_variant: dict[str, dict[str, dict[str, Any]]]
    cv_scores: dict[str, dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"by_variant": self.by_variant, "cv_scores": self.cv_scores}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TunedParams":
        return cls(data["by_variant"], data.get("cv_scores", {}))


@dataclass
class ExperimentResult:
    params: TunedParams
    models: dict[str, tuple[Pipeline, Pipeline]]  # variant -> (regressor, classifier)
    served_variant: str
    report: EvaluationReport

    @property
    def regressor(self) -> Pipeline:
        return self.models[self.served_variant][0]

    @property
    def classifier(self) -> Pipeline:
        return self.models[self.served_variant][1]


def make_splits(flights: pd.DataFrame, config: dict[str, Any], audit: CleaningAudit | None = None) -> Splits:
    frame = make_model_frame(flights, config, audit)
    eval_cfg = config["evaluation"]
    return time_split(frame, eval_cfg["time_column"], eval_cfg["train_end"], eval_cfg["val_end"])


def _targets(df: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.Series, pd.Series]:
    pipeline_cfg = config["pipeline"]
    return df[pipeline_cfg["target_regression"]], df[pipeline_cfg["target_classification"]]


def _default_params(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    pipeline_cfg = config["pipeline"]
    return {"regression": dict(pipeline_cfg["lgbm_params"]), "classification": dict(pipeline_cfg["logistic_params"])}


def tune_models(splits: Splits, config: dict[str, Any], enabled: bool | None = None) -> TunedParams:
    """Tune each variant on train+val with rolling-origin CV. The test split is not used."""
    pipeline_cfg = config["pipeline"]
    tuning = pipeline_cfg["tuning"]
    variants = list(config["features"]["variants"])
    if not (tuning["enabled"] if enabled is None else enabled):
        logger.info("Tuning disabled; using default parameters from config.yaml")
        return TunedParams({v: _default_params(config) for v in variants})

    cv_splits = config["evaluation"]["cv_splits"]
    y_reg, y_cls = _targets(splits.train_val, config)
    by_variant, cv_scores = {}, {}
    for variant in variants:
        features = variant_feature_set(config, variant)
        X = splits.train_val[features.columns]
        reg, reg_cv = tune_regression_hyperparameters(
            X, y_reg, pipeline_cfg, features, tuning["n_trials_regression"], cv_splits
        )
        cls, cls_cv = tune_classification_hyperparameters(
            X, y_cls, pipeline_cfg, features, tuning["n_trials_classification"], cv_splits
        )
        by_variant[variant] = {"regression": reg, "classification": cls}
        cv_scores[variant] = {"cv_mae": reg_cv, "cv_brier": cls_cv}
    return TunedParams(by_variant, cv_scores)


def _fit_predict(
    config: dict[str, Any],
    features: FeatureSet,
    params: dict[str, dict[str, Any]],
    fit: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[Pipeline, Pipeline, np.ndarray, tuple[np.ndarray, np.ndarray]]:
    pipeline_cfg = config["pipeline"]
    y_reg, y_cls = _targets(fit, config)
    X_fit, X_test = fit[features.columns], test[features.columns]
    regressor = build_regression_pipeline(pipeline_cfg, params["regression"], features).fit(X_fit, y_reg)
    classifier = build_classification_pipeline(pipeline_cfg, params["classification"], features).fit(X_fit, y_cls)
    return (
        regressor,
        classifier,
        regressor.predict(X_test),
        (classifier.predict(X_test), classifier.predict_proba(X_test)[:, 1]),
    )


def train_and_evaluate(
    splits: Splits, params: TunedParams, config: dict[str, Any], audit: CleaningAudit | None = None
) -> ExperimentResult:
    """Fit models and baselines on train+val, then score everything once on test."""
    pipeline_cfg, eval_cfg = config["pipeline"], config["evaluation"]
    fit, test = splits.train_val, splits.test
    y_reg_fit, y_cls_fit = _targets(fit, config)
    y_reg_test, y_cls_test = (y.to_numpy() for y in _targets(test, config))
    indices = bootstrap_indices(len(test), eval_cfg["bootstrap_samples"], pipeline_cfg["random_state"])
    confidence = eval_cfg["confidence_level"]
    threshold = pipeline_cfg["delay_threshold_minutes"]

    reg_predictions: dict[str, np.ndarray] = {
        name: GroupMedianRegressor(columns).fit(fit, y_reg_fit).predict(test)
        for name, columns in REGRESSION_BASELINES.items()
    }
    cls_predictions: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, columns in CLASSIFICATION_BASELINES.items():
        baseline = GroupRateClassifier(columns).fit(fit, y_cls_fit)
        cls_predictions[name] = (baseline.predict(test), baseline.predict_proba(test)[:, 1])

    models = {}
    for variant in config["features"]["variants"]:
        features = variant_feature_set(config, variant)
        regressor, classifier, reg_pred, cls_pred = _fit_predict(
            config, features, params.by_variant[variant], fit, test
        )
        models[variant] = (regressor, classifier)
        reg_predictions[regressor_name(variant)] = reg_pred
        cls_predictions[classifier_name(variant)] = cls_pred

    report = EvaluationReport(
        split_summary=splits.summary(eval_cfg["time_column"]),
        tasks=[
            evaluate_regression(y_reg_test, reg_predictions, list(REGRESSION_BASELINES), indices, confidence),
            evaluate_classification(
                y_cls_test, cls_predictions, list(CLASSIFICATION_BASELINES), indices, confidence, threshold
            ),
        ],
        bootstrap_samples=eval_cfg["bootstrap_samples"],
        confidence=confidence,
        extra_sections=[
            *_ablation_section(splits, config, indices, confidence),
            *_coverage_section(splits, config),
            *_tuning_section(params),
            *_audit_section(audit),
        ],
    )
    return ExperimentResult(params, models, config["features"]["served_variant"], report)


def _ablation_section(splits: Splits, config: dict[str, Any], indices: np.ndarray, confidence: float) -> list[str]:
    """Add feature groups one at a time, with default parameters so only the features change."""
    reg_preds: dict[str, np.ndarray] = {}
    cls_preds: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for label, features in ablation_steps(config):
        _, _, reg_preds[label], cls_preds[label] = _fit_predict(
            config, features, _default_params(config), splits.train_val, splits.test
        )
    y_reg, y_cls = (y.to_numpy() for y in _targets(splits.test, config))
    first = [next(iter(reg_preds))]
    regression = evaluate_regression(y_reg, reg_preds, first, indices, confidence)
    classification = evaluate_classification(
        y_cls, cls_preds, first, indices, confidence, config["pipeline"]["delay_threshold_minutes"]
    )
    return [
        "## Feature ablation",
        "",
        "Each row adds one feature group to the row above (LightGBM for minutes, logistic regression for "
        "the classifier, default parameters). Changes are paired differences against the first row; "
        "negative is better for MAE and Brier.",
        "",
        ablation_markdown(regression, classification),
        "",
    ]


def _coverage_section(splits: Splits, config: dict[str, Any]) -> list[str]:
    groups = config["features"]["groups"]
    rows = ["| Group | Feature | Known in train+val | Known in test |", "| --- | --- | --- | --- |"]
    for group in ("rotation", "live_propagation", "rates"):
        for feature in groups.get(group, {}).get("numerical", []):
            features = FeatureSet((feature,), ())
            fit_cov = feature_coverage(splits.train_val, features).iloc[0]
            test_cov = feature_coverage(splits.test, features).iloc[0]
            rows.append(f"| {group} | `{feature}` | {fit_cov:.0%} | {test_cov:.0%} |")
    return [
        "## Feature coverage",
        "",
        "Aircraft registrations are published only for Mehrabad departures, so rotation and live features "
        "exist mostly there. Inbound landing times are missing from the 2025 scrapes, so the inbound delay "
        "features are empty until newly scraped data fills them. Rates are empty for the first days of data.",
        "",
        *rows,
        "",
    ]


def _tuning_section(params: TunedParams) -> list[str]:
    if not params.cv_scores:
        return ["## Tuning", "", "Tuning was disabled; default parameters from `config.yaml` were used.", ""]
    lines = [
        "## Tuning",
        "",
        "Parameters were chosen by rolling-origin cross-validation on train+val (test unused):",
        "",
    ]
    for variant, scores in params.cv_scores.items():
        lines.append(
            f"- {VARIANT_LABELS.get(variant, variant)}: CV MAE {scores['cv_mae']:.1f} min, "
            f"CV Brier {scores['cv_brier']:.3f}"
        )
    return [*lines, ""]


def _audit_section(audit: CleaningAudit | None) -> list[str]:
    if audit is None or not audit.steps:
        return []
    return ["## Data cleaning", "", "Rows remaining after each cleaning step:", "", audit.to_markdown(), ""]


def run_experiment(
    flights: pd.DataFrame, config: dict[str, Any], tune: bool | None = None, audit: CleaningAudit | None = None
) -> ExperimentResult:
    splits = make_splits(flights, config, audit)
    params = tune_models(splits, config, enabled=tune)
    return train_and_evaluate(splits, params, config, audit)


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

    audit = CleaningAudit()
    with tempfile.TemporaryDirectory() as tmp:
        cleaned_path = Path(tmp) / "cleaned.csv"
        # Only the cleaning steps go in the audit here; make_splits records the training-row steps
        audit.steps = clean_flights_file(resolve_path(args.raw), cleaned_path).steps[:3]
        flights = read_flights(cleaned_path)

    result = run_experiment(flights, config, tune=args.tune, audit=audit)
    output = resolve_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(result.report.to_markdown(), encoding="utf-8")
    print(result.report.to_markdown())
    logger.info("Report written to %s", output)


if __name__ == "__main__":
    main()
