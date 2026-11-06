from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pytest
from mlflow.exceptions import MlflowException

from aeroscrape.features.preprocess import FeatureSet
from aeroscrape.models.delay_model import DelayModelBundle, ProbabilityCalibrator, log_delay_model
from aeroscrape.models.experiment import choose_threshold, conformal_margin
from aeroscrape.models.train import build_lgbm_classification_pipeline, build_quantile_regressor

FEATURES = FeatureSet(("hour", "prev_delay"), ("airline",))
CONFIG = {"random_state": 0}
SMALL = {"n_estimators": 30, "learning_rate": 0.1, "num_leaves": 7, "min_child_samples": 5, "verbose": -1}


@pytest.fixture(scope="module")
def data() -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(0)
    n = 400
    X = pd.DataFrame(
        {
            "hour": rng.integers(0, 24, n),
            "prev_delay": np.where(rng.random(n) < 0.3, np.nan, rng.gamma(2, 15, n)),
            "airline": rng.choice(["A", "B", "C"], n),
        }
    )
    y = 10 + np.nan_to_num(X["prev_delay"].to_numpy()) * 0.8 + rng.gamma(2, 8, n)
    return X, y


@pytest.fixture(scope="module")
def bundle(data) -> DelayModelBundle:
    X, y = data
    fit, cal = X.iloc[:300], X.iloc[300:]
    quantiles = {
        q: build_quantile_regressor(CONFIG, SMALL, FEATURES, q, "log").fit(fit, y[:300]) for q in (0.1, 0.5, 0.9)
    }
    raw = build_lgbm_classification_pipeline(CONFIG, SMALL | {"objective": "binary"}, FEATURES).fit(fit, y[:300] > 15)
    calibrator = ProbabilityCalibrator("sigmoid").fit(raw.predict_proba(cal)[:, 1], y[300:] > 15)
    return DelayModelBundle("booking", FEATURES, quantiles, 3.0, "log", raw, calibrator, threshold=0.6)


def test_range_is_ordered_and_widened(bundle: DelayModelBundle, data) -> None:
    X, _ = data
    low_raw, mid_raw, high_raw = bundle.predict_range(X, conformal=False)
    low, mid, high = bundle.predict_range(X)

    assert (low <= mid).all() and (mid <= high).all()
    assert np.allclose(mid, mid_raw)
    assert np.allclose(high - low, (high_raw - low_raw) + 6)


def test_predict_output(bundle: DelayModelBundle, data) -> None:
    out = bundle.predict(data[0].head(10))
    assert list(out.columns) == ["p_delay_over_15", "likely_delayed", "p10", "p50", "p90"]
    assert out["p_delay_over_15"].between(0, 1).all()
    assert (out["likely_delayed"] == (out["p_delay_over_15"] >= 0.6)).all()


def test_explanations_name_input_features(bundle: DelayModelBundle, data) -> None:
    factors = bundle.explain(data[0].head(3), top=3)
    assert len(factors) == 3
    assert {f["feature"] for row in factors for f in row} <= set(FEATURES.columns)
    # One-hot columns are summed back: one value per input feature
    assert bundle.shap_values(data[0].head(3)).shape == (3, 3)


def test_pyfunc_round_trip_enforces_the_signature(bundle: DelayModelBundle, data, tmp_path) -> None:
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    with mlflow.start_run():
        info = log_delay_model(bundle, data[0].head(20))
    loaded = mlflow.pyfunc.load_model(info.model_uri)

    X = bundle.prepare_input(data[0].head(5))
    pd.testing.assert_frame_equal(loaded.predict(X), bundle.predict(X))
    with pytest.raises(MlflowException):
        loaded.predict(X.drop(columns=["airline"]))


def test_threshold_is_the_lowest_meeting_the_precision_target() -> None:
    proba = np.linspace(0, 1, 200)
    y = (proba > 0.4).astype(int)  # everything above 0.4 is delayed
    threshold, reached = choose_threshold(y, proba, precision_target=0.75)
    assert reached
    assert y[proba >= threshold].mean() >= 0.75
    lower = proba[proba < threshold].max()
    assert y[proba >= lower].mean() < 0.75  # any lower threshold misses the target


def test_threshold_falls_back_when_the_target_is_unreachable() -> None:
    rng = np.random.default_rng(1)
    assert choose_threshold(rng.integers(0, 2, 300), rng.random(300), 0.99) == (0.5, False)


def test_conformal_margin_reaches_the_target_coverage() -> None:
    rng = np.random.default_rng(2)
    y = rng.normal(0, 1, 2000)
    low, high = np.full_like(y, -0.5), np.full_like(y, 0.5)  # too narrow: covers ~38%
    margin = conformal_margin(y[:1000], low[:1000], high[:1000], coverage=0.8)

    covered = ((y[1000:] >= low[1000:] - margin) & (y[1000:] <= high[1000:] + margin)).mean()
    assert margin > 0
    assert 0.77 <= covered <= 0.83


@pytest.mark.parametrize("method", ["sigmoid", "isotonic"])
def test_calibrator_fixes_overconfident_probabilities(method: str) -> None:
    rng = np.random.default_rng(3)
    true_p = rng.uniform(0.2, 0.8, 4000)
    y = (rng.random(4000) < true_p).astype(int)
    overconfident = np.clip(0.5 + (true_p - 0.5) * 1.8, 0, 1)

    calibrator = ProbabilityCalibrator(method).fit(overconfident[:2000], y[:2000])
    fixed = calibrator.transform(overconfident[2000:])

    def brier(p):
        return np.mean((p - y[2000:]) ** 2)

    assert brier(fixed) < brier(overconfident[2000:])


def test_saved_bundle_does_not_need_dill(bundle: DelayModelBundle, data, tmp_path) -> None:
    import io
    import pickletools

    import dill

    dill.extend(True)  # as in Airflow's task runtime, where models are trained
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    with mlflow.start_run():
        info = log_delay_model(bundle, data[0].head(5))
    saved = Path(mlflow.artifacts.download_artifacts(info.model_uri)) / "artifacts" / "bundle.pkl"

    names = {a for _, a, _ in pickletools.genops(io.BytesIO(saved.read_bytes())) if isinstance(a, str)}
    assert not any("dill" in n for n in names)
