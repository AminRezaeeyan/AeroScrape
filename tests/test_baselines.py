import numpy as np
import pandas as pd

from aeroscrape.models.baselines import GroupMedianRegressor, GroupRateClassifier

X = pd.DataFrame({"airline": ["A", "A", "A", "B", "B"], "slot": ["am", "am", "pm", "am", "am"]})
DELAYS = np.array([10.0, 30.0, 100.0, 0.0, 4.0])
LATE = np.array([1, 1, 1, 0, 0])


def test_group_median_with_fallback_for_unseen_groups() -> None:
    model = GroupMedianRegressor(["airline", "slot"]).fit(X, DELAYS)
    unseen = pd.DataFrame({"airline": ["A", "C"], "slot": ["am", "am"]})

    assert model.predict(unseen).tolist() == [20.0, 10.0]  # group median, then global median


def test_global_median_without_groups() -> None:
    assert GroupMedianRegressor([]).fit(X, DELAYS).predict(X).tolist() == [10.0] * 5


def test_group_rate_probabilities_and_labels() -> None:
    model = GroupRateClassifier(["airline"]).fit(X, LATE)
    unseen = pd.DataFrame({"airline": ["A", "B", "C"]})

    assert model.predict_proba(unseen)[:, 1].tolist() == [1.0, 0.0, 0.6]
    assert model.predict(unseen).tolist() == [1, 0, 1]


def test_majority_baseline_predicts_the_majority_class() -> None:
    model = GroupRateClassifier([]).fit(X, LATE)
    assert model.predict(X).tolist() == [1] * 5
