"""The served model: a calibrated delay probability and a delay range, as one MLflow artifact.

`DelayModelBundle` holds everything a prediction needs, so the API never pairs a
model from the registry with a preprocessor from disk. `DelayModel` wraps it as an
MLflow pyfunc with a signature, which makes MLflow reject malformed input.
"""

import logging
import pickle
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import mlflow
import numpy as np
import pandas as pd
import shap
from mlflow.models import infer_signature
from mlflow.pyfunc import PythonModel
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from aeroscrape.features.preprocess import FeatureSet, prepare_input

logger = logging.getLogger(__name__)

OUTPUT_COLUMNS = ["p_delay_over_15", "likely_delayed", "p10", "p50", "p90"]


class ProbabilityCalibrator:
    """Maps raw classifier probabilities to calibrated ones.

    Fit on out-of-fold predictions, so it learns how the model's probabilities
    compare with outcomes on flights it was not trained on.
    """

    def __init__(self, method: str = "sigmoid"):
        if method not in ("sigmoid", "isotonic"):
            raise ValueError(f"Unknown calibration method {method!r}")
        self.method = method

    def fit(self, proba: np.ndarray, y: np.ndarray) -> "ProbabilityCalibrator":
        if self.method == "sigmoid":
            # Platt scaling: a one-feature logistic regression on the log-odds
            self.model_ = LogisticRegression(C=1e6).fit(_logit(proba)[:, None], y)
        else:
            self.model_ = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(proba, y)
        return self

    def transform(self, proba: np.ndarray) -> np.ndarray:
        if self.method == "sigmoid":
            return self.model_.predict_proba(_logit(proba)[:, None])[:, 1]
        return self.model_.predict(proba)


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


@dataclass
class DelayModelBundle:
    variant: str
    features: FeatureSet
    quantile_models: dict[float, Any]  # 0.1, 0.5, 0.9 -> fitted regressor
    conformal_margin: float  # minutes added on both sides of the 10-90% range
    target_transform: str
    raw_classifier: Any  # LightGBM classifier pipeline; also explained by SHAP
    calibrator: ProbabilityCalibrator
    threshold: float  # calibrated probability at which a flight is "likely delayed"
    metadata: dict[str, Any] = field(default_factory=dict)

    def prepare_input(self, X: pd.DataFrame) -> pd.DataFrame:
        return prepare_input(X, self.features)

    def known_categories(self) -> dict[str, set[str]]:
        """The values of each categorical feature the model was trained on; others are ignored by it."""
        preprocess = self.raw_classifier.named_steps["preprocess"]
        encoder = preprocess.named_transformers_["cat"].named_steps["onehot"]
        return {
            col: {str(v) for v in values}
            for col, values in zip(self.features.categorical, encoder.categories_, strict=True)
        }

    def predict_range(self, X: pd.DataFrame, conformal: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        X = self.prepare_input(X)
        raw = np.column_stack([self.quantile_models[q].predict(X) for q in (0.1, 0.5, 0.9)])
        # Separately trained quantile models can cross; sorting each row restores p10 <= p50 <= p90
        low, mid, high = np.sort(raw, axis=1).T
        if conformal:
            # A negative margin narrows an interval that was too wide, but never past the median
            low = np.minimum(low - self.conformal_margin, mid)
            high = np.maximum(high + self.conformal_margin, mid)
        return low, mid, high

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        raw = self.raw_classifier.predict_proba(self.prepare_input(X))[:, 1]
        return self.calibrator.transform(raw)

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        low, mid, high = self.predict_range(X)
        p = self.predict_proba(X)
        return pd.DataFrame(
            {
                "p_delay_over_15": p,
                "likely_delayed": p >= self.threshold,
                "p10": low,
                "p50": mid,
                "p90": high,
            },
            index=X.index,
        )

    def shap_values(self, X: pd.DataFrame) -> pd.DataFrame:
        """SHAP contributions of the uncalibrated LightGBM classifier per input feature, in log-odds.

        Calibration is a monotone map applied afterwards, so it changes how big the probability
        is but not which features push it up or down. One-hot columns and missing-value
        indicators are summed back into the input feature they came from.
        """
        X = self.prepare_input(X)
        preprocess, model = self.raw_classifier.named_steps["preprocess"], self.raw_classifier.named_steps["model"]
        values = shap.TreeExplainer(model).shap_values(preprocess.transform(X))
        if isinstance(values, list):  # older SHAP returns one array per class
            values = values[1]
        values = np.asarray(values)
        if values.ndim == 3:
            values = values[:, :, 1]
        owners = _input_feature_of(preprocess.get_feature_names_out(), self.features)
        return pd.DataFrame(values, columns=owners, index=X.index).T.groupby(level=0).sum().T

    def explain(self, X: pd.DataFrame, top: int = 8) -> list[list[dict[str, Any]]]:
        """The largest contributions for each row, with the feature's value, in log-odds."""
        contributions = self.shap_values(X)
        inputs = self.prepare_input(X)
        explanations = []
        for row in range(len(X)):
            ranked = contributions.iloc[row].abs().sort_values(ascending=False).index[:top]
            explanations.append(
                [
                    {
                        "feature": name,
                        "value": _display(inputs.iloc[row][name]),
                        "contribution": round(float(contributions.iloc[row][name]), 3),
                    }
                    for name in ranked
                ]
            )
        return explanations


def _display(value: Any) -> str:
    if isinstance(value, float) and np.isnan(value):
        return "unknown"
    return f"{value:g}" if isinstance(value, float) else str(value)


def _input_feature_of(transformed_names: np.ndarray, features: FeatureSet) -> list[str]:
    """Map ColumnTransformer output names (num__x, num__missingindicator_x, cat__x_value) to input columns."""
    by_length = sorted(features.columns, key=len, reverse=True)
    owners = []
    for name in transformed_names:
        _, _, rest = str(name).partition("__")
        rest = rest.removeprefix("missingindicator_")
        owners.append(next((col for col in by_length if rest == col or rest.startswith(col + "_")), rest))
    return owners


class DelayModel(PythonModel):
    """MLflow wrapper; the bundle itself is stored as a pickle artifact."""

    def load_context(self, context) -> None:
        path = context.artifacts["bundle"]
        if path.endswith(".joblib"):  # versions registered before the switch to plain pickle
            self.bundle: DelayModelBundle = joblib.load(path)
        else:
            with open(path, "rb") as f:
                self.bundle = pickle.load(f)

    def predict(self, context, model_input: pd.DataFrame, params: dict[str, Any] | None = None) -> pd.DataFrame:
        return self.bundle.predict(model_input)


def log_delay_model(bundle: DelayModelBundle, example: pd.DataFrame, registered_model_name: str | None = None):
    """Log the bundle as a pyfunc model with a signature and input example; optionally register it."""
    example = bundle.prepare_input(example)
    with tempfile.TemporaryDirectory() as tmp:
        # Written with the C pickler: Airflow loads dill, which installs its own savers in Python's
        # pure-Python pickler (the one joblib uses), so models saved by joblib in a training task
        # needed dill to load. The C pickler doesn't use those savers.
        path = Path(tmp) / "bundle.pkl"
        with open(path, "wb") as f:
            pickle.dump(bundle, f, protocol=5)
        return mlflow.pyfunc.log_model(
            name="delay-model",
            python_model=DelayModel(),
            artifacts={"bundle": str(path)},
            signature=infer_signature(example, bundle.predict(example)),
            input_example=example.head(5),
            registered_model_name=registered_model_name,
            pip_requirements=[],  # the serving image installs aeroscrape and its locked dependencies
        )
