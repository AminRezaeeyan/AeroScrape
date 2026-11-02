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


REGRESSION_METRICS = (
    MetricSpec("MAE (min)", _mae, lower_is_better=True, fmt="{:.1f}"),
    MetricSpec("MedAE (min)", _medae, lower_is_better=True, fmt="{:.1f}"),
    MetricSpec("Within 15 min", _within_15, lower_is_better=False, fmt="{:.1%}"),
)
CLASSIFICATION_METRICS = (
    MetricSpec("Accuracy", _accuracy, lower_is_better=False, fmt="{:.3f}"),
    MetricSpec("PR-AUC", _pr_auc, lower_is_better=False, fmt="{:.3f}"),
    MetricSpec("Brier", _brier, lower_is_better=True, fmt="{:.3f}"),
)


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
    primary: MetricSpec,
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
    sign = 1 if primary.lower_is_better else -1
    best_baseline = min(baselines, key=lambda b: sign * results[b][primary.name].point)
    comparisons = []
    for name in candidates:
        diff_draws = draws[name][primary.name] - draws[best_baseline][primary.name]
        diff_point = results[name][primary.name].point - results[best_baseline][primary.name].point
        comparisons.append(Comparison(name, best_baseline, primary, _interval(diff_point, diff_draws, confidence)))
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
        REGRESSION_METRICS[0],
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
        CLASSIFICATION_METRICS[2],
        indices,
        confidence,
        notes=[
            f"{y.mean():.1%} of test flights are delayed, so always predicting 'delayed' "
            f"scores {max(y.mean(), 1 - y.mean()):.3f} accuracy. PR-AUC of a random ranking equals that rate."
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
