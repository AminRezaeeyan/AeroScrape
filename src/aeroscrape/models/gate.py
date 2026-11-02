"""Champion/challenger gate: a newly trained model replaces the served one only if it is clearly better.

Without a gate every training run went straight to production, so a model trained on a bad
day of data, or with a bug, replaced a good one. The gate scores both models on the same
newest held-out flights and promotes the challenger only when all checks in
`config.yaml` (`gate`) pass:

1. enough held-out flights to judge (`min_gate_rows`);
2. the challenger's MAE is lower than the champion's by more than bootstrap noise
   (the upper end of the paired 95% interval of the difference is below zero);
3. the challenger beats the airline x time-of-day baseline the same way;
4. its Brier score is not worse than the champion's by more than `max_brier_increase`;
5. its 10-90% range covers between `coverage_range` of the flights.

With no champion yet, checks 2 and 4 are skipped.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from aeroscrape.models.evaluate import bootstrap_indices


@dataclass
class ModelPredictions:
    name: str
    p50: np.ndarray
    p10: np.ndarray | None = None
    p90: np.ndarray | None = None
    p_delay: np.ndarray | None = None


@dataclass
class Check:
    name: str
    passed: bool
    detail: str


@dataclass
class GateResult:
    promote: bool
    checks: list[Check]
    metrics: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def decision(self) -> str:
        return "promote" if self.promote else "reject"

    def to_markdown(self, champion: str, challenger: str) -> str:
        lines = [
            f"# Promotion gate: {challenger} vs {champion}",
            "",
            f"**Decision: {self.decision}.**",
            "",
            "| Check | Result | Detail |",
            "| --- | --- | --- |",
            *[f"| {c.name} | {'pass' if c.passed else 'FAIL'} | {c.detail} |" for c in self.checks],
            "",
            "| Metric | Value |",
            "| --- | --- |",
            *[f"| {k} | {v:.4g} |" for k, v in self.metrics.items()],
        ]
        if self.notes:
            lines += ["", *[f"- {n}" for n in self.notes]]
        return "\n".join(lines) + "\n"


def _mae(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y - pred)))


def paired_mae_difference(
    y: np.ndarray, a: np.ndarray, b: np.ndarray, indices: np.ndarray, confidence: float
) -> tuple[float, float, float]:
    """MAE(a) - MAE(b) with a paired bootstrap interval: (point, low, high)."""
    err_a, err_b = np.abs(y - a), np.abs(y - b)
    draws = err_a[indices].mean(axis=1) - err_b[indices].mean(axis=1)
    tail = (1 - confidence) / 2 * 100
    low, high = np.percentile(draws, [tail, 100 - tail])
    return float(err_a.mean() - err_b.mean()), float(low), float(high)


def evaluate_gate(
    y_minutes: np.ndarray,
    y_delayed: np.ndarray,
    challenger: ModelPredictions,
    champion: ModelPredictions | None,
    baseline: np.ndarray,
    gate_config: dict[str, Any],
    bootstrap_samples: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> GateResult:
    y_minutes = np.asarray(y_minutes, dtype=float)
    n = len(y_minutes)
    checks: list[Check] = []
    metrics: dict[str, float] = {"gate_rows": n, "challenger_mae": _mae(y_minutes, challenger.p50)}

    min_rows = gate_config["min_gate_rows"]
    checks.append(Check("Enough held-out flights", n >= min_rows, f"{n} flights (minimum {min_rows})"))
    if n < min_rows:
        return GateResult(False, checks, metrics)

    indices = bootstrap_indices(n, bootstrap_samples, seed)
    pct = f"{confidence:.0%}"

    if champion is not None:
        point, low, high = paired_mae_difference(y_minutes, challenger.p50, champion.p50, indices, confidence)
        metrics |= {"champion_mae": _mae(y_minutes, champion.p50), "mae_change_vs_champion": point}
        checks.append(
            Check(
                "MAE better than champion beyond noise",
                high < 0,
                f"change {point:+.2f} min, {pct} interval [{low:+.2f}, {high:+.2f}]",
            )
        )

    point, low, high = paired_mae_difference(y_minutes, challenger.p50, baseline, indices, confidence)
    metrics |= {"baseline_mae": _mae(y_minutes, baseline), "mae_change_vs_baseline": point}
    checks.append(
        Check(
            "MAE better than airline x time-of-day baseline",
            high < 0,
            f"change {point:+.2f} min, {pct} interval [{low:+.2f}, {high:+.2f}]",
        )
    )

    if champion is not None and challenger.p_delay is not None and champion.p_delay is not None:
        y_delayed = np.asarray(y_delayed, dtype=float)
        brier_challenger = float(np.mean((challenger.p_delay - y_delayed) ** 2))
        brier_champion = float(np.mean((champion.p_delay - y_delayed) ** 2))
        increase = brier_challenger - brier_champion
        limit = gate_config["max_brier_increase"]
        metrics |= {"challenger_brier": brier_challenger, "champion_brier": brier_champion}
        checks.append(
            Check(
                "Calibration not worse than champion",
                increase <= limit,
                f"Brier {brier_challenger:.4f} vs {brier_champion:.4f} (allowed increase {limit})",
            )
        )

    if challenger.p10 is not None and challenger.p90 is not None:
        coverage = float(np.mean((y_minutes >= challenger.p10) & (y_minutes <= challenger.p90)))
        low_cov, high_cov = gate_config["coverage_range"]
        metrics["challenger_coverage"] = coverage
        checks.append(
            Check(
                "10-90% range coverage in range",
                low_cov <= coverage <= high_cov,
                f"{coverage:.1%} (allowed {low_cov:.0%} to {high_cov:.0%})",
            )
        )

    notes = [] if champion is not None else ["No champion yet: the challenger only has to beat the baseline."]
    return GateResult(all(c.passed for c in checks), checks, metrics, notes)
