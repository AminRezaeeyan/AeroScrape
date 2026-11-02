"""Simple reference models. A model is only useful if it beats these on the same test set."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin


def _group_keys(X: pd.DataFrame, columns: list[str]) -> pd.Series:
    return X[columns].astype(str).agg("|".join, axis=1)


class GroupQuantileRegressor(RegressorMixin, BaseEstimator):
    """Predict a training quantile of the row's group, or the global quantile for unseen groups.

    With `group_columns=[]` this is a global baseline, e.g. the global median for quantile 0.5.
    """

    def __init__(self, group_columns: list[str] | None = None, quantile: float = 0.5):
        self.group_columns = group_columns
        self.quantile = quantile

    def fit(self, X: pd.DataFrame, y) -> "GroupQuantileRegressor":
        y = pd.Series(np.asarray(y, dtype=float), index=X.index)
        self.global_value_ = float(y.quantile(self.quantile))
        columns = list(self.group_columns or [])
        self.group_values_ = y.groupby(_group_keys(X, columns)).quantile(self.quantile).to_dict() if columns else {}
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        columns = list(self.group_columns or [])
        if not columns:
            return np.full(len(X), self.global_value_)
        keys = _group_keys(X, columns)
        return keys.map(self.group_values_).fillna(self.global_value_).to_numpy(dtype=float)


class GroupMedianRegressor(GroupQuantileRegressor):
    """The group median; see GroupQuantileRegressor."""

    def __init__(self, group_columns: list[str] | None = None):
        super().__init__(group_columns, quantile=0.5)


class GroupRateClassifier(ClassifierMixin, BaseEstimator):
    """Predict the training positive rate of the row's group as the probability.

    With `group_columns=[]` it predicts the overall rate, i.e. always the majority class.
    """

    def __init__(self, group_columns: list[str] | None = None, threshold: float = 0.5):
        self.group_columns = group_columns
        self.threshold = threshold

    def fit(self, X: pd.DataFrame, y) -> "GroupRateClassifier":
        y = pd.Series(np.asarray(y, dtype=float), index=X.index)
        self.classes_ = np.array([0, 1])
        self.global_rate_ = float(y.mean())
        columns = list(self.group_columns or [])
        self.group_rates_ = y.groupby(_group_keys(X, columns)).mean().to_dict() if columns else {}
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        columns = list(self.group_columns or [])
        if columns:
            p = _group_keys(X, columns).map(self.group_rates_).fillna(self.global_rate_).to_numpy(dtype=float)
        else:
            p = np.full(len(X), self.global_rate_)
        return np.column_stack([1 - p, p])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        # Ties at exactly the threshold go to the positive class, so the overall-rate
        # baseline predicts the majority class even when the rate is 0.5.
        return (self.predict_proba(X)[:, 1] >= self.threshold).astype(int)
