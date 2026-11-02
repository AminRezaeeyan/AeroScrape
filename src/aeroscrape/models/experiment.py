"""Split, tune, train, calibrate and evaluate, reusable from Airflow and from the command line.

Train and validation are pooled and split again by rolling-origin CV (each fold trains on
earlier flights and predicts later ones). The out-of-fold predictions, which span about
twelve days, are used to:

- tune hyperparameters,
- choose the target transform,
- calibrate the probabilities and choose the decision threshold,
- widen the delay range so it covers ~80% of flights (conformal prediction).

Calibrating on the 2-day validation window alone was tried first: that window had 52%
delayed flights against 60% in train and test, and the calibrated probabilities came out
too low on test. The final models are refit on all of train+val; test is scored once.

Run locally without Airflow or MLflow:

    python -m aeroscrape.models.experiment            # tune, train, evaluate, write the report
    python -m aeroscrape.models.experiment --no-tune  # use the default parameters from config.yaml
"""

import argparse
import logging
import math
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline

from aeroscrape.config import get_app_config, resolve_path
from aeroscrape.data.clean import CleaningAudit, clean_flights_file, read_flights
from aeroscrape.data.split import Splits, time_split
from aeroscrape.features.preprocess import (
    FeatureSet,
    ablation_steps,
    feature_coverage,
    make_model_frame,
    prepare_input,
    variant_feature_set,
)
from aeroscrape.models.baselines import GroupMedianRegressor, GroupQuantileRegressor, GroupRateClassifier
from aeroscrape.models.delay_model import DelayModelBundle, ProbabilityCalibrator
from aeroscrape.models.evaluate import (
    EvaluationReport,
    ablation_markdown,
    bootstrap_indices,
    evaluate_classification,
    evaluate_intervals,
    evaluate_regression,
    expected_calibration_error,
)
from aeroscrape.models.plots import calibration_bins, importance_figure, reliability_figure
from aeroscrape.models.train import (
    build_classification_pipeline,
    build_lgbm_classification_pipeline,
    build_quantile_regressor,
    build_regression_pipeline,
)
from aeroscrape.models.tune import (
    tune_classification_hyperparameters,
    tune_lgbm_classification_hyperparameters,
    tune_regression_hyperparameters,
)

logger = logging.getLogger(__name__)

VARIANT_LABELS = {"booking": "at booking", "pre_departure": "1 h before departure"}
# Baselines see only the columns named here, fit on train+val (more data than the models get)
REGRESSION_BASELINES = {
    "Global median": [],
    "Airline × time-of-day median": ["airline", "scheduled_time_of_day"],
}
INTERVAL_BASELINES = {
    "Global percentiles": [],
    "Airline × time-of-day percentiles": ["airline", "scheduled_time_of_day"],
}
CLASSIFICATION_BASELINES = {
    "Majority class": [],
    "Airline delay rate": ["airline"],
}
MIN_FLAGGED_FOR_THRESHOLD = 20  # don't trust a precision measured on fewer flights


def _label(variant: str) -> str:
    return VARIANT_LABELS.get(variant, variant)


def regressor_name(variant: str) -> str:
    return f"LightGBM median ({_label(variant)})"


def classifier_name(variant: str) -> str:
    return f"LightGBM, calibrated ({_label(variant)})"


@dataclass
class TunedParams:
    """Per variant: {"regression": ..., "classification": ..., "lgbm_classification": ...} plus CV scores."""

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
    bundles: dict[str, DelayModelBundle]
    served_variant: str
    report: EvaluationReport
    figures: dict[str, Figure] = field(default_factory=dict)

    @property
    def served(self) -> DelayModelBundle:
        return self.bundles[self.served_variant]


def make_splits(flights: pd.DataFrame, config: dict[str, Any], audit: CleaningAudit | None = None) -> Splits:
    frame = make_model_frame(flights, config, audit)
    eval_cfg = config["evaluation"]
    return time_split(frame, eval_cfg["time_column"], eval_cfg["train_end"], eval_cfg["val_end"])


def _targets(df: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.Series, pd.Series]:
    pipeline_cfg = config["pipeline"]
    return df[pipeline_cfg["target_regression"]], df[pipeline_cfg["target_classification"]]


def _default_params(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    pipeline_cfg = config["pipeline"]
    return {
        "regression": dict(pipeline_cfg["lgbm_params"]),
        "classification": dict(pipeline_cfg["logistic_params"]),
        "lgbm_classification": dict(pipeline_cfg["lgbm_classifier_params"]),
    }


def tune_models(splits: Splits, config: dict[str, Any], enabled: bool | None = None) -> TunedParams:
    """Tune each variant with rolling-origin CV on train+val. The test split is not used."""
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
        X = prepare_input(splits.train_val, features)
        reg, reg_cv = tune_regression_hyperparameters(
            X, y_reg, pipeline_cfg, features, tuning["n_trials_regression"], cv_splits
        )
        cls, cls_cv = tune_classification_hyperparameters(
            X, y_cls, pipeline_cfg, features, tuning["n_trials_classification"], cv_splits
        )
        lgbm_cls, lgbm_cls_cv = tune_lgbm_classification_hyperparameters(
            X, y_cls, pipeline_cfg, features, tuning["n_trials_lgbm_classification"], cv_splits
        )
        by_variant[variant] = {"regression": reg, "classification": cls, "lgbm_classification": lgbm_cls}
        cv_scores[variant] = {"cv_mae": reg_cv, "cv_brier_logistic": cls_cv, "cv_brier_lgbm": lgbm_cls_cv}
    return TunedParams(by_variant, cv_scores)


# ---------------------------------------------------------------------------- fitting one variant


def choose_threshold(y: np.ndarray, proba: np.ndarray, precision_target: float) -> tuple[float, bool]:
    """The lowest threshold whose validation precision reaches the target (so recall is highest).

    Returns (threshold, reached). If no threshold flagging at least MIN_FLAGGED_FOR_THRESHOLD
    flights reaches the target, falls back to 0.5.
    """
    for t in np.unique(np.round(proba, 4)):
        flagged = proba >= t
        if flagged.sum() < MIN_FLAGGED_FOR_THRESHOLD:
            break
        if y[flagged].mean() >= precision_target:
            return float(t), True
    return 0.5, False


def conformal_margin(y: np.ndarray, low: np.ndarray, high: np.ndarray, coverage: float) -> float:
    """Conformalized quantile regression: how much to widen [low, high] so it covers `coverage` of new flights."""
    scores = np.maximum(low - y, y - high)
    n = len(scores)
    level = min(1.0, math.ceil((n + 1) * coverage) / n)
    return float(np.quantile(scores, level, method="higher"))


@dataclass
class VariantFit:
    bundle: DelayModelBundle
    transform_oof_mae: dict[str, float]
    oof_precision: float
    oof_recall: float
    threshold_reached: bool
    uncalibrated: Pipeline
    logistic: Pipeline


def fit_variant(variant: str, params: dict[str, dict[str, Any]], splits: Splits, config: dict[str, Any]) -> VariantFit:
    pipeline_cfg, eval_cfg = config["pipeline"], config["evaluation"]
    features = variant_feature_set(config, variant)
    data = splits.train_val
    X = prepare_input(data, features)
    y_reg, y_cls = (y.to_numpy() for y in _targets(data, config))
    folds = list(TimeSeriesSplit(n_splits=eval_cfg["cv_splits"]).split(X))
    oof = np.concatenate([test_idx for _, test_idx in folds])

    def out_of_fold(build) -> np.ndarray:
        """Predictions for every out-of-fold row, each from a model fit only on earlier flights."""
        predictions = np.empty(len(X))
        for train_idx, test_idx in folds:
            model, target = build()
            model.fit(X.iloc[train_idx], target[train_idx])
            predictions[test_idx] = (
                model.predict_proba(X.iloc[test_idx])[:, 1]
                if hasattr(model, "predict_proba")
                else model.predict(X.iloc[test_idx])
            )
        return predictions[oof]

    def quantile(q: float, transform: str):
        return lambda: (build_quantile_regressor(pipeline_cfg, params["regression"], features, q, transform), y_reg)

    # Target transform: keep whichever gives the lowest out-of-fold MAE for the median
    medians = {t: out_of_fold(quantile(0.5, t)) for t in pipeline_cfg["target_transforms"]}
    transform_mae = {t: float(np.mean(np.abs(y_reg[oof] - pred))) for t, pred in medians.items()}
    # Prefer the plain target unless a transform is clearly better; ties are noise
    best = min(transform_mae, key=lambda t: transform_mae[t])
    plain = transform_mae.get("none", float("inf"))
    transform = best if transform_mae[best] < plain * (1 - pipeline_cfg["transform_min_gain"]) else "none"
    oof_range = np.sort(
        np.column_stack(
            [out_of_fold(quantile(0.1, transform)), medians[transform], out_of_fold(quantile(0.9, transform))]
        ),
        axis=1,
    )
    raw_oof = out_of_fold(
        lambda: (build_lgbm_classification_pipeline(pipeline_cfg, params["lgbm_classification"], features), y_cls)
    )
    calibrator = ProbabilityCalibrator(pipeline_cfg["calibration_method"]).fit(raw_oof, y_cls[oof])
    p_oof = calibrator.transform(raw_oof)

    # Final models on all of train+val
    quantile_models = {q: quantile(q, transform)()[0].fit(X, y_reg) for q in (0.1, 0.5, 0.9)}
    uncalibrated = build_lgbm_classification_pipeline(pipeline_cfg, params["lgbm_classification"], features).fit(
        X, y_cls
    )
    logistic = build_classification_pipeline(pipeline_cfg, params["classification"], features).fit(X, y_cls)

    margin = 0.0
    if pipeline_cfg["conformal_intervals"]:
        margin = conformal_margin(y_reg[oof], oof_range[:, 0], oof_range[:, 2], pipeline_cfg["interval_coverage"])
    threshold, reached = choose_threshold(y_cls[oof], p_oof, pipeline_cfg["precision_target"])
    flagged = p_oof >= threshold

    bundle = DelayModelBundle(
        variant=variant,
        features=features,
        quantile_models=quantile_models,
        conformal_margin=margin,
        target_transform=transform,
        raw_classifier=uncalibrated,
        calibrator=calibrator,
        threshold=threshold,
        metadata={
            "variant": variant,
            "trained_on": [str(data[eval_cfg["time_column"]].min()), str(data[eval_cfg["time_column"]].max())],
            "train_rows": len(data),
            "out_of_fold_rows": len(oof),
            "target_transform": transform,
            "conformal_margin_min": margin,
            "calibration_method": pipeline_cfg["calibration_method"],
            "threshold": threshold,
            "precision_target": pipeline_cfg["precision_target"],
            "delay_threshold_minutes": pipeline_cfg["delay_threshold_minutes"],
            "features": features.columns,
        },
    )
    return VariantFit(
        bundle,
        transform_mae,
        float(y_cls[oof][flagged].mean()) if flagged.any() else float("nan"),
        float(flagged[y_cls[oof] == 1].mean()),
        reached,
        uncalibrated,
        logistic,
    )


# ---------------------------------------------------------------------------- evaluation


def _fit_predict(
    config: dict[str, Any],
    features: FeatureSet,
    params: dict[str, dict[str, Any]],
    fit: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[Pipeline, Pipeline, np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """Point regressor and logistic classifier on one feature set; used by the ablation."""
    pipeline_cfg = config["pipeline"]
    y_reg, y_cls = _targets(fit, config)
    X_fit, X_test = prepare_input(fit, features), prepare_input(test, features)
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
    pipeline_cfg, eval_cfg = config["pipeline"], config["evaluation"]
    fit, test = splits.train_val, splits.test
    y_reg_fit, y_cls_fit = _targets(fit, config)
    y_reg_test, y_cls_test = (y.to_numpy() for y in _targets(test, config))
    indices = bootstrap_indices(len(test), eval_cfg["bootstrap_samples"], pipeline_cfg["random_state"])
    confidence = eval_cfg["confidence_level"]

    point: dict[str, np.ndarray] = {
        name: GroupMedianRegressor(cols).fit(fit, y_reg_fit).predict(test)
        for name, cols in REGRESSION_BASELINES.items()
    }
    ranges: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {
        name: tuple(  # type: ignore[misc]
            GroupQuantileRegressor(cols, q).fit(fit, y_reg_fit).predict(test) for q in (0.1, 0.5, 0.9)
        )
        for name, cols in INTERVAL_BASELINES.items()
    }
    probabilities: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, cols in CLASSIFICATION_BASELINES.items():
        baseline = GroupRateClassifier(cols).fit(fit, y_cls_fit)
        probabilities[name] = (baseline.predict(test), baseline.predict_proba(test)[:, 1])

    fits = {
        variant: fit_variant(variant, params.by_variant[variant], splits, config)
        for variant in config["features"]["variants"]
    }
    for variant, vf in fits.items():
        bundle, label = vf.bundle, _label(variant)
        X_test = prepare_input(test, bundle.features)
        low, mid, high = bundle.predict_range(X_test, conformal=False)
        point[regressor_name(variant)] = mid
        ranges[f"LightGBM percentiles ({label})"] = (low, mid, high)
        ranges[f"LightGBM percentiles + conformal ({label})"] = bundle.predict_range(X_test)
        raw_p = vf.uncalibrated.predict_proba(X_test)[:, 1]
        cal_p = bundle.predict_proba(X_test)
        probabilities[f"Logistic regression ({label})"] = (
            vf.logistic.predict(X_test),
            vf.logistic.predict_proba(X_test)[:, 1],
        )
        probabilities[f"LightGBM, uncalibrated ({label})"] = ((raw_p >= 0.5).astype(int), raw_p)
        probabilities[classifier_name(variant)] = ((cal_p >= 0.5).astype(int), cal_p)

    served = config["features"]["served_variant"]
    figures: dict[str, Figure] = {}
    sections = [
        *_threshold_section(fits, test, config, indices, confidence),
        *_calibration_section(fits[served], test, config, figures),
        *_error_analysis_section(fits, served, test, config),
        *_importance_section(fits[served], test, figures),
        *_ablation_section(splits, config, indices, confidence),
        *_coverage_section(splits, config),
        *_choices_section(fits, params),
        *_audit_section(audit),
    ]
    report = EvaluationReport(
        split_summary=splits.summary(eval_cfg["time_column"]),
        tasks=[
            evaluate_regression(y_reg_test, point, list(REGRESSION_BASELINES), indices, confidence),
            evaluate_intervals(y_reg_test, ranges, list(INTERVAL_BASELINES), indices, confidence),
            evaluate_classification(
                y_cls_test,
                probabilities,
                list(CLASSIFICATION_BASELINES),
                indices,
                confidence,
                pipeline_cfg["delay_threshold_minutes"],
            ),
        ],
        bootstrap_samples=eval_cfg["bootstrap_samples"],
        confidence=confidence,
        extra_sections=sections,
    )
    # Raw percentiles and uncalibrated probabilities are shown for context, not compared
    for task in report.tasks:
        task.comparisons = [c for c in task.comparisons if not _is_diagnostic(c.model)]
    report.tasks[2].notes.append(
        "Accuracy, precision and recall here use a 0.5 cutoff for every model; the chosen decision "
        "threshold is evaluated in its own section below."
    )
    return ExperimentResult(params, {v: f.bundle for v, f in fits.items()}, served, report, figures)


def _is_diagnostic(name: str) -> bool:
    return "uncalibrated" in name or (name.startswith("LightGBM percentiles (") and "conformal" not in name)


def _threshold_section(
    fits: dict[str, VariantFit], test: pd.DataFrame, config: dict[str, Any], indices: np.ndarray, confidence: float
) -> list[str]:
    target = config["pipeline"]["precision_target"]
    y = _targets(test, config)[1].to_numpy()
    lines = [
        "## Decision threshold",
        "",
        f"The API flags a flight as likely delayed when its calibrated probability reaches a threshold chosen "
        f"on the out-of-fold predictions as the lowest one with precision of at least {target:.0%} (so recall "
        "is as high as the precision target allows).",
        "",
        "| Model | Threshold | Out-of-fold precision | Out-of-fold recall | Test precision | Test recall | Flagged in test |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for variant, vf in fits.items():
        p = vf.bundle.predict_proba(prepare_input(test, vf.bundle.features))
        flagged = (p >= vf.bundle.threshold).astype(int)
        task = evaluate_classification(y, {"m": (flagged, p)}, ["m"], indices, confidence, 15)
        result = task.results["m"]
        note = "" if vf.threshold_reached else " (target not reached; 0.5 used)"
        lines.append(
            f"| {classifier_name(variant)} | {vf.bundle.threshold:.2f}{note} | {vf.oof_precision:.3f} | "
            f"{vf.oof_recall:.3f} | {result['Precision'].format('{:.3f}')} | {result['Recall'].format('{:.3f}')} | "
            f"{flagged.mean():.0%} |"
        )
    return [*lines, ""]


def _calibration_section(
    vf: VariantFit, test: pd.DataFrame, config: dict[str, Any], figures: dict[str, Figure]
) -> list[str]:
    bundle = vf.bundle
    X = prepare_input(test, bundle.features)
    y = _targets(test, config)[1].to_numpy()
    curves = {
        "LightGBM, calibrated": bundle.predict_proba(X),
        "LightGBM, uncalibrated": vf.uncalibrated.predict_proba(X)[:, 1],
        "Logistic regression": vf.logistic.predict_proba(X)[:, 1],
    }
    label = _label(bundle.variant)
    figures["reliability"] = reliability_figure(y, curves, f"Reliability on test ({label})")
    lines = [
        "## Calibration",
        "",
        f"Test flights grouped by predicted probability ({label}). For a calibrated model the observed share "
        f"delayed matches the predicted probability. Calibration is {config['pipeline']['calibration_method']} "
        f"scaling fit on {bundle.metadata['out_of_fold_rows']} out-of-fold predictions from rolling-origin CV "
        "over train+val. Chart: `reliability.png`.",
        "",
        "| Predicted (mean) | Observed | Flights | | Model |",
        "| --- | --- | --- | --- | --- |",
    ]
    for name, proba in curves.items():
        for predicted, observed, n in calibration_bins(y, proba):
            lines.append(f"| {predicted:.2f} | {observed:.2f} | {n} | | {name} |")
    lines += [
        "",
        "Expected calibration error on test: "
        + ", ".join(f"{name} {expected_calibration_error(y, p):.3f}" for name, p in curves.items())
        + ".",
    ]
    return [*lines, ""]


def _segment(test: pd.DataFrame) -> dict[str, pd.Series]:
    airports = test["airport"].where(test["airport"].isin(["فرودگاه مهرآباد", "فرودگاه مشهد"]), "other airports")
    top_airlines = test["airline"].value_counts().index[:5]
    return {
        "Airport": airports,
        "Time of day": test["scheduled_time_of_day"],
        "Airline": test["airline"].where(test["airline"].isin(top_airlines), "other airlines"),
        "Inbound flight linked": test["has_inbound"].map({1: "yes", 0: "no"}),
        "Previous delay of aircraft known": test["prev_departure_delay_min"].notna().map({True: "yes", False: "no"}),
    }


def _error_analysis_section(
    fits: dict[str, VariantFit], served: str, test: pd.DataFrame, config: dict[str, Any]
) -> list[str]:
    y_reg, y_cls = (y.to_numpy() for y in _targets(test, config))
    preds = {}
    for variant, vf in fits.items():
        X = prepare_input(test, vf.bundle.features)
        low, mid, high = vf.bundle.predict_range(X)
        preds[variant] = (
            np.abs(y_reg - mid),
            (y_reg >= low) & (y_reg <= high),
            (vf.bundle.predict_proba(X) - y_cls) ** 2,
        )
    other = next((v for v in fits if v != served), served)
    lines = [
        "## Error analysis",
        "",
        f"Test errors by segment. MAE and Brier for the model {_label(served)}, MAE also for the model "
        f"{_label(other)}; coverage is of the conformal 10-90% range. Segments are small, so differences "
        "of a few minutes are within noise.",
        "",
        f"| Segment | Value | Flights | Delayed | MAE ({_label(served)}) | MAE ({_label(other)}) | Coverage | Brier |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for segment, values in _segment(test).items():
        for value in values.value_counts().index:
            mask = (values == value).to_numpy()
            abs_err, covered, sq_err = preds[served]
            lines.append(
                f"| {segment} | {value} | {mask.sum()} | {y_cls[mask].mean():.0%} | {abs_err[mask].mean():.1f} | "
                f"{preds[other][0][mask].mean():.1f} | {covered[mask].mean():.0%} | {sq_err[mask].mean():.3f} |"
            )
    return [*lines, ""]


def _importance_section(vf: VariantFit, test: pd.DataFrame, figures: dict[str, Figure]) -> list[str]:
    shap_values = vf.bundle.shap_values(test)
    importance = shap_values.abs().mean().sort_values(ascending=False).head(12)
    figures["shap_importance"] = importance_figure(
        list(importance.index),
        list(importance.to_numpy()),
        f"What drives the delay probability ({_label(vf.bundle.variant)})",
        "Mean |SHAP value|, log-odds of a delay over 15 min",
    )
    rows = [f"| `{name}` | {value:.3f} |" for name, value in importance.items()]
    return [
        "## What drives the predictions",
        "",
        "Mean absolute SHAP value of the uncalibrated LightGBM classifier on test, in log-odds of a delay over "
        "15 minutes, summed over each feature's one-hot columns. Calibration is monotone, so it does not change "
        "the ranking. Chart: `shap_importance.png`.",
        "",
        "| Feature | Mean \\|SHAP\\| |",
        "| --- | --- |",
        *rows,
        "",
    ]


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
        "Each row adds one feature group to the row above (LightGBM median regression for minutes, logistic "
        "regression for the classifier, default parameters, fit on train+val). Changes are paired differences "
        "against the first row; negative is better for MAE and Brier.",
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


def _choices_section(fits: dict[str, VariantFit], params: TunedParams) -> list[str]:
    lines = ["## Choices made from out-of-fold predictions", ""]
    for variant, vf in fits.items():
        b = vf.bundle
        maes = ", ".join(f"{t} {mae:.2f}" for t, mae in vf.transform_oof_mae.items())
        lines.append(
            f"- {_label(variant)}: target transform **{b.target_transform}** (out-of-fold MAE: {maes}); "
            f"conformal margin {b.conformal_margin:+.1f} min; threshold {b.threshold:.2f}."
        )
    if params.cv_scores:
        lines += ["", "Hyperparameters were chosen by rolling-origin cross-validation on train+val:", ""]
        for variant, scores in params.cv_scores.items():
            lines.append(
                f"- {_label(variant)}: CV MAE {scores['cv_mae']:.1f} min, CV Brier {scores['cv_brier_lgbm']:.3f} "
                f"(LightGBM) and {scores['cv_brier_logistic']:.3f} (logistic regression)"
            )
    else:
        lines += ["", "Tuning was disabled; default parameters from `config.yaml` were used."]
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


def save_outputs(result: ExperimentResult, report_path: Path) -> None:
    """Write the markdown report and its charts next to it."""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(result.report.to_markdown(), encoding="utf-8")
    for name, figure in result.figures.items():
        figure.savefig(report_path.parent / f"{name}.png", facecolor=figure.get_facecolor())


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    config = get_app_config()
    parser = argparse.ArgumentParser(description="Train, calibrate and evaluate the delay models on a time split.")
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
    save_outputs(result, output)
    print(result.report.to_markdown())
    logger.info("Report and charts written to %s", output.parent)


if __name__ == "__main__":
    main()
