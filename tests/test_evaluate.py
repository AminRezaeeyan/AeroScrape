import numpy as np

from aeroscrape.models.evaluate import bootstrap_indices, evaluate_classification, evaluate_regression

rng = np.random.default_rng(0)
Y = rng.gamma(2, 15, size=300)


def _regression(predictions: dict[str, np.ndarray]):
    return evaluate_regression(Y, predictions, ["Baseline"], bootstrap_indices(len(Y), 300, seed=1), 0.95)


def test_intervals_contain_the_point_estimate() -> None:
    report = _regression({"Baseline": np.full_like(Y, np.median(Y)), "Model": Y + rng.normal(0, 5, len(Y))})
    for estimates in report.results.values():
        for est in estimates.values():
            assert est.low <= est.point <= est.high


def test_clearly_better_model_is_reported_as_better() -> None:
    report = _regression({"Baseline": np.full_like(Y, np.median(Y)), "Model": Y + rng.normal(0, 2, len(Y))})
    comparison = report.comparisons[0]

    assert comparison.difference.high < 0
    assert "better" in comparison.verdict


def test_identical_model_shows_no_evidence_of_difference() -> None:
    baseline = np.full_like(Y, np.median(Y))
    report = _regression({"Baseline": baseline, "Model": baseline + 1e-9})
    assert "no evidence" in report.comparisons[0].verdict


def test_bootstrap_is_reproducible() -> None:
    assert (bootstrap_indices(50, 10, seed=3) == bootstrap_indices(50, 10, seed=3)).all()


def test_constant_probability_has_pr_auc_equal_to_the_positive_rate() -> None:
    y = (rng.random(400) < 0.7).astype(int)
    p = np.full(len(y), 0.7)
    report = evaluate_classification(
        y, {"Majority": (np.ones_like(y), p)}, ["Majority"], bootstrap_indices(len(y), 50, seed=1), 0.95, 15
    )
    assert abs(report.results["Majority"]["PR-AUC"].point - y.mean()) < 1e-9


def test_undefined_metric_is_shown_as_not_available() -> None:
    from aeroscrape.models.evaluate import Estimate

    assert Estimate(float("nan"), float("nan"), float("nan")).format("{:.3f}") == "n/a"
