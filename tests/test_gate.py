import numpy as np
import pytest

from aeroscrape.models.gate import ModelPredictions, evaluate_gate

GATE = {"min_gate_rows": 200, "max_brier_increase": 0.005, "coverage_range": [0.70, 0.90]}
rng = np.random.default_rng(0)
N = 600
Y = rng.gamma(2, 15, N)
DELAYED = (Y > 15).astype(int)
BASELINE = np.full(N, np.median(Y))


def _model(noise: float, width: float | None = None, p_noise: float = 0.15) -> ModelPredictions:
    # +-1.28 standard deviations of the error covers about 80% of flights
    width = 1.28 * noise if width is None else width
    p50 = Y + rng.normal(0, noise, N)
    p = np.clip(DELAYED * 0.7 + 0.15 + rng.normal(0, p_noise, N), 0.01, 0.99)
    return ModelPredictions("m", p50, p50 - width, p50 + width, p)


def _check(result, name: str) -> bool:
    return next(c.passed for c in result.checks if c.name.startswith(name))


def test_clearly_better_challenger_is_promoted() -> None:
    result = evaluate_gate(Y, DELAYED, _model(5), _model(15), BASELINE, GATE, 300)
    assert result.promote, result.to_markdown("champion", "challenger")


def test_worse_challenger_is_rejected() -> None:
    result = evaluate_gate(Y, DELAYED, _model(15), _model(5), BASELINE, GATE, 300)
    assert not result.promote
    assert not _check(result, "MAE better than champion")


def test_equal_models_keep_the_champion() -> None:
    champion = _model(8)
    challenger = ModelPredictions("same", champion.p50 + 1e-9, champion.p10, champion.p90, champion.p_delay)
    result = evaluate_gate(Y, DELAYED, challenger, champion, BASELINE, GATE, 300)
    assert not result.promote  # an improvement within noise is not an improvement


def test_too_few_flights_rejects_without_comparing() -> None:
    model = _model(1)
    small = ModelPredictions("m", model.p50[:50], model.p10[:50], model.p90[:50], model.p_delay[:50])
    result = evaluate_gate(Y[:50], DELAYED[:50], small, None, BASELINE[:50], GATE, 100)
    assert not result.promote
    assert [c.name for c in result.checks] == ["Enough held-out flights"]


def test_badly_calibrated_challenger_is_rejected() -> None:
    champion = _model(15, p_noise=0.05)
    challenger = _model(5, p_noise=0.45)
    result = evaluate_gate(Y, DELAYED, challenger, champion, BASELINE, GATE, 300)
    assert _check(result, "MAE better than champion")
    assert not _check(result, "Calibration not worse")
    assert not result.promote


@pytest.mark.parametrize("width", [1.0, 200.0])
def test_range_that_covers_too_little_or_too_much_is_rejected(width: float) -> None:
    result = evaluate_gate(Y, DELAYED, _model(5, width=width), None, BASELINE, GATE, 300)
    assert not _check(result, "10-90% range coverage")
    assert not result.promote


def test_first_model_only_needs_to_beat_the_baseline() -> None:
    result = evaluate_gate(Y, DELAYED, _model(5), None, BASELINE, GATE, 300)
    assert result.promote
    assert "No champion" in result.to_markdown("no champion", "challenger")
