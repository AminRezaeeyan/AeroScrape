"""Test-set evaluation with baselines and bootstrap confidence intervals.

Every model is scored on the same bootstrap resamples of the test set, so the
difference between a model and a baseline is estimated as a paired difference.
With a few hundred test flights, a one-minute gain in MAE can easily be noise,
and the interval on that difference is what tells us whether it is.
"""

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, brier_score_loss

from aeroscrape.data.split import SplitSummary

# Regression metrics take (y_true, y_pred); classification metrics take (y_true, labels, probabilities)
RegressionMetric = Callable[[np.ndarray, np.ndarray], float]
ClassificationMetric = Callable[[np.ndarray, np.ndarray, np.ndarray], float]


@dataclass(frozen=True)
class MetricSpec:
    name: str
    fn: Callable[..., float]
    lower_is_better: bool
    fmt: str


def _mae(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y - pred)))


def _medae(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.median(np.abs(y - pred)))


def _within_15(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y - pred) <= 15))


def _accuracy(y: np.ndarray, labels: np.ndarray, proba: np.ndarray) -> float:
    return float(accuracy_score(y, labels))


def _pr_auc(y: np.ndarray, labels: np.ndarray, proba: np.ndarray) -> float:
    # Undefined when a resample happens to contain a single class
    return float(average_precision_score(y, proba)) if 0 < y.sum() < len(y) else float("nan")


def _brier(y: np.ndarray, labels: np.ndarray, proba: np.ndarray) -> float:
    return float(brier_score_loss(y, proba))


def _precision(y: np.ndarray, labels: np.ndarray, proba: np.ndarray) -> float:
    predicted = labels == 1
    return float(np.mean(y[predicted])) if predicted.any() else float("nan")


def _recall(y: np.ndarray, labels: np.ndarray, proba: np.ndarray) -> float:
    actual = y == 1
    return float(np.mean(labels[actual] == 1)) if actual.any() else float("nan")


def expected_calibration_error(y: np.ndarray, proba: np.ndarray, bins: int = 10) -> float:
    """Average gap between predicted probability and observed rate, over equal-width bins, weighted by size."""
    which = np.minimum((proba * bins).astype(int), bins - 1)
    gaps = [abs(proba[which == b].mean() - y[which == b].mean()) * np.mean(which == b) for b in np.unique(which)]
    return float(np.sum(gaps))


def _ece(y: np.ndarray, labels: np.ndarray, proba: np.ndarray) -> float:
    return expected_calibration_error(y, proba)


QUANTILES = (0.1, 0.5, 0.9)


def pinball_loss(y: np.ndarray, prediction: np.ndarray, quantile: float) -> float:
    diff = y - prediction
    return float(np.mean(np.maximum(quantile * diff, (quantile - 1) * diff)))


def _mean_pinball(y: np.ndarray, low: np.ndarray, mid: np.ndarray, high: np.ndarray) -> float:
    return float(np.mean([pinball_loss(y, q, tau) for q, tau in zip((low, mid, high), QUANTILES, strict=True)]))


def _coverage(y: np.ndarray, low: np.ndarray, mid: np.ndarray, high: np.ndarray) -> float:
    return float(np.mean((y >= low) & (y <= high)))


def _width(y: np.ndarray, low: np.ndarray, mid: np.ndarray, high: np.ndarray) -> float:
    return float(np.mean(high - low))


REGRESSION_METRICS = (
    MetricSpec("MAE (min)", _mae, lower_is_better=True, fmt="{:.1f}"),
    MetricSpec("MedAE (min)", _medae, lower_is_better=True, fmt="{:.1f}"),
    MetricSpec("Within 15 min", _within_15, lower_is_better=False, fmt="{:.1%}"),
)
CLASSIFICATION_METRICS = (
    MetricSpec("Accuracy", _accuracy, lower_is_better=False, fmt="{:.3f}"),
    MetricSpec("Precision", _precision, lower_is_better=False, fmt="{:.3f}"),
    MetricSpec("Recall", _recall, lower_is_better=False, fmt="{:.3f}"),
    MetricSpec("PR-AUC", _pr_auc, lower_is_better=False, fmt="{:.3f}"),
    MetricSpec("Brier", _brier, lower_is_better=True, fmt="{:.3f}"),
    MetricSpec("ECE", _ece, lower_is_better=True, fmt="{:.3f}"),
)
# Coverage has a target (the nominal 80%), not a direction; comparisons use pinball loss
INTERVAL_METRICS = (
    MetricSpec("Coverage of 10-90% interval", _coverage, lower_is_better=False, fmt="{:.1%}"),
    MetricSpec("Mean width (min)", _width, lower_is_better=True, fmt="{:.1f}"),
    MetricSpec("Pinball loss (min)", _mean_pinball, lower_is_better=True, fmt="{:.2f}"),
)


def metric(metrics: tuple[MetricSpec, ...], name: str) -> MetricSpec:
    return next(m for m in metrics if m.name == name)


@dataclass(frozen=True)
class Estimate:
    point: float
    low: float
    high: float

    def format(self, fmt: str) -> str:
        return f"{fmt.format(self.point)} [{fmt.format(self.low)}, {fmt.format(self.high)}]"


@dataclass
class Comparison:
    model: str
    baseline: str
    metric: MetricSpec
    difference: Estimate  # model minus baseline

    @property
    def verdict(self) -> str:
        if self.difference.low <= 0 <= self.difference.high:
            return "the interval includes 0, so this test set gives no evidence of a difference"
        better = (self.difference.high < 0) == self.metric.lower_is_better
        return f"the interval excludes 0, so the model is {'better' if better else 'worse'}"


@dataclass
class TaskReport:
    title: str
    metrics: tuple[MetricSpec, ...]
    baselines: list[str]
    candidates: list[str]
    results: dict[str, dict[str, Estimate]]
    comparisons: list[Comparison]
    notes: list[str] = field(default_factory=list)

    def to_markdown(self) -> str:
        header = "| Model | " + " | ".join(m.name for m in self.metrics) + " |"
        lines = [f"## {self.title}", "", *self.notes, "", header, "|" + " --- |" * (len(self.metrics) + 1)]
        for name in self.baselines + self.candidates:
            label = f"{name} (baseline)" if name in self.baselines else f"**{name}**"
            cells = [self.results[name][m.name].format(m.fmt) for m in self.metrics]
            lines.append(f"| {label} | " + " | ".join(cells) + " |")
        for c in self.comparisons:
            lines += [
                "",
                f"**{c.model} minus best baseline ({c.baseline}), {c.metric.name}:** "
                f"{c.difference.format(c.metric.fmt)}. "
                f"{c.verdict[0].upper()}{c.verdict[1:]}.",
            ]
        return "\n".join(lines)


def bootstrap_indices(n: int, samples: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n, size=(samples, n))


def _interval(point: float, draws: np.ndarray, confidence: float) -> Estimate:
    tail = (1 - confidence) / 2 * 100
    low, high = np.nanpercentile(draws, [tail, 100 - tail])
    return Estimate(point, float(low), float(high))


def _evaluate(
    title: str,
    metrics: tuple[MetricSpec, ...],
    y: np.ndarray,
    predictions: Mapping[str, tuple[np.ndarray, ...]],
    baselines: list[str],
    compare: tuple[MetricSpec, ...],
    indices: np.ndarray,
    confidence: float,
    notes: list[str],
) -> TaskReport:
    draws: dict[str, dict[str, np.ndarray]] = {}
    results: dict[str, dict[str, Estimate]] = {}
    for name, preds in predictions.items():
        draws[name], results[name] = {}, {}
        for metric in metrics:
            point = metric.fn(y, *preds)
            resampled = np.array([metric.fn(y[i], *(p[i] for p in preds)) for i in indices])
            draws[name][metric.name] = resampled
            results[name][metric.name] = _interval(point, resampled, confidence)

    candidates = [name for name in predictions if name not in baselines]
    comparisons = []
    for metric in compare:
        sign = 1 if metric.lower_is_better else -1
        best_baseline = min(baselines, key=lambda b: sign * results[b][metric.name].point)
        for name in candidates:
            diff_draws = draws[name][metric.name] - draws[best_baseline][metric.name]
            diff_point = results[name][metric.name].point - results[best_baseline][metric.name].point
            comparisons.append(Comparison(name, best_baseline, metric, _interval(diff_point, diff_draws, confidence)))
    return TaskReport(title, metrics, baselines, candidates, results, comparisons, notes)


def evaluate_regression(
    y: np.ndarray,
    predictions: Mapping[str, np.ndarray],
    baselines: list[str],
    indices: np.ndarray,
    confidence: float,
) -> TaskReport:
    y = np.asarray(y, dtype=float)
    return _evaluate(
        "Delay in minutes (regression)",
        REGRESSION_METRICS,
        y,
        {name: (np.asarray(p, dtype=float),) for name, p in predictions.items()},
        baselines,
        (REGRESSION_METRICS[0],),
        indices,
        confidence,
        notes=[f"Test delays: median {np.median(y):.0f} min, mean {np.mean(y):.0f} min, max {np.max(y):.0f} min."],
    )


def evaluate_classification(
    y: np.ndarray,
    predictions: Mapping[str, tuple[np.ndarray, np.ndarray]],
    baselines: list[str],
    indices: np.ndarray,
    confidence: float,
    threshold_minutes: float,
) -> TaskReport:
    y = np.asarray(y, dtype=int)
    return _evaluate(
        f"Delayed more than {threshold_minutes:g} minutes (classification)",
        CLASSIFICATION_METRICS,
        y,
        {name: (np.asarray(labels), np.asarray(proba, dtype=float)) for name, (labels, proba) in predictions.items()},
        baselines,
        (metric(CLASSIFICATION_METRICS, "Brier"), metric(CLASSIFICATION_METRICS, "PR-AUC")),
        indices,
        confidence,
        notes=[
            f"{y.mean():.1%} of test flights are delayed, so always predicting 'delayed' "
            f"scores {max(y.mean(), 1 - y.mean()):.3f} accuracy. PR-AUC of a random ranking equals that rate."
        ],
    )


def ablation_markdown(regression: TaskReport, classification: TaskReport) -> str:
    """One row per cumulative feature set; differences are paired against the first row."""

    def diff(task: TaskReport, model: str, metric: str) -> str:
        for c in task.comparisons:
            if c.model == model and c.metric.name == metric:
                return c.difference.format("{:+.1f}" if "min" in metric else "{:+.3f}")
        return "reference"

    mae = REGRESSION_METRICS[0]
    brier, pr_auc = metric(CLASSIFICATION_METRICS, "Brier"), metric(CLASSIFICATION_METRICS, "PR-AUC")
    lines = [
        f"| Features | {mae.name} | Change in MAE | {brier.name} | Change in Brier | {pr_auc.name} |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for step in regression.baselines + regression.candidates:
        lines.append(
            f"| {step} | {regression.results[step][mae.name].format(mae.fmt)} | {diff(regression, step, mae.name)} "
            f"| {classification.results[step][brier.name].format(brier.fmt)} | {diff(classification, step, brier.name)} "
            f"| {classification.results[step][pr_auc.name].format(pr_auc.fmt)} |"
        )
    return "\n".join(lines)


def evaluate_intervals(
    y: np.ndarray,
    predictions: Mapping[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    baselines: list[str],
    indices: np.ndarray,
    confidence: float,
) -> TaskReport:
    """Score (10th, 50th, 90th percentile) predictions; a calibrated interval covers about 80% of flights."""
    y = np.asarray(y, dtype=float)
    return _evaluate(
        "Delay range: 10th to 90th percentile",
        INTERVAL_METRICS,
        y,
        {name: tuple(np.asarray(q, dtype=float) for q in qs) for name, qs in predictions.items()},
        baselines,
        (metric(INTERVAL_METRICS, "Pinball loss (min)"),),
        indices,
        confidence,
        notes=[
            "A calibrated 10-90% interval contains about 80% of actual delays. Pinball loss scores all three "
            "percentiles at once and rewards narrow intervals that still cover."
        ],
    )


@dataclass
class EvaluationReport:
    split_summary: dict[str, SplitSummary]
    tasks: list[TaskReport]
    bootstrap_samples: int
    confidence: float
    extra_sections: list[str] = field(default_factory=list)

    def to_markdown(self) -> str:
        test = self.split_summary["test"]
        split_rows = [
            f"| {name} | {s.rows} | {s.start:%Y-%m-%d %H:%M} | {s.end:%Y-%m-%d %H:%M} |"
            for name, s in self.split_summary.items()
        ]
        parts = [
            "# Evaluation report",
            "",
            f"Test window: {test.start:%Y-%m-%d} to {test.end:%Y-%m-%d} "
            f"({test.rows} flights). Models and baselines are fit on train+val and scored once on test. "
            f"Brackets are {self.confidence:.0%} bootstrap percentile intervals over {self.bootstrap_samples:,} "
            "resamples of the test set; differences are paired, using the same resamples for both models.",
            "",
            "## Data",
            "",
            "| Split | Rows | From | To |",
            "| --- | --- | --- | --- |",
            *split_rows,
            "",
        ]
        for task in self.tasks:
            parts += [task.to_markdown(), ""]
        parts += self.extra_sections
        return "\n".join(parts).rstrip() + "\n"

    def to_metrics(self) -> dict[str, float]:
        """Flat metric names for MLflow, e.g. 'test/lightgbm/mae_min' and its _low/_high bounds."""
        flat: dict[str, float] = {}
        for task in self.tasks:
            for model, estimates in task.results.items():
                for metric_name, est in estimates.items():
                    key = f"test/{_slug(model)}/{_slug(metric_name)}"
                    flat |= {key: est.point, f"{key}_low": est.low, f"{key}_high": est.high}
            for c in task.comparisons:
                key = f"test/{_slug(c.model)}/{_slug(c.metric.name)}_minus_best_baseline"
                flat |= {key: c.difference.point, f"{key}_low": c.difference.low, f"{key}_high": c.difference.high}
        return flat

    def to_dict(self) -> dict[str, Any]:
        return {
            "bootstrap_samples": self.bootstrap_samples,
            "confidence": self.confidence,
            "splits": {
                name: {"rows": sp.rows, "from": str(sp.start), "to": str(sp.end)}
                for name, sp in self.split_summary.items()
            },
            "metrics": self.to_metrics(),
        }


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
