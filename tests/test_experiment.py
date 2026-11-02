import copy
from pathlib import Path

import pandas as pd
import pytest

from aeroscrape.config import get_app_config
from aeroscrape.data.clean import clean_flights_file, read_flights
from aeroscrape.data.split import Splits
from aeroscrape.models.experiment import make_splits, regressor_name, run_experiment, tune_models


@pytest.fixture(scope="module")
def cleaned(tmp_path_factory: pytest.TempPathFactory) -> pd.DataFrame:
    out = tmp_path_factory.mktemp("clean") / "cleaned.csv"
    clean_flights_file(Path(__file__).parent / "fixtures" / "sample_raw.csv", out)
    return read_flights(out)


@pytest.fixture
def config() -> dict:
    cfg = copy.deepcopy(get_app_config())
    cfg["evaluation"] |= {"bootstrap_samples": 50, "cv_splits": 2}
    cfg["pipeline"]["tuning"] |= {
        "n_trials_regression": 2,
        "n_trials_classification": 2,
        "n_trials_lgbm_classification": 2,
    }
    cfg["pipeline"]["lgbm_params"] |= {"n_estimators": 20, "min_child_samples": 2}
    cfg["pipeline"]["lgbm_classifier_params"] |= {"n_estimators": 20, "min_child_samples": 2}
    # The 200-row fixture has few late days; a wider validation window gives calibration both classes
    cfg["evaluation"] |= {"train_end": "2025-06-02", "val_end": "2025-06-10"}
    return cfg


def test_experiment_runs_end_to_end(cleaned: pd.DataFrame, config: dict) -> None:
    result = run_experiment(cleaned, config, tune=True)
    report = result.report.to_markdown()

    assert regressor_name("booking") in report and regressor_name("pre_departure") in report
    assert "## Feature ablation" in report and "## Decision threshold" in report and "## Error analysis" in report
    assert result.params.cv_scores["booking"].keys() == {"cv_mae", "cv_brier_logistic", "cv_brier_lgbm"}
    assert set(result.figures) == {"reliability", "shap_importance"}
    prediction = result.served.predict(result.served.prepare_input(make_splits(cleaned, config).test))
    assert (prediction["p10"] <= prediction["p50"]).all() and (prediction["p50"] <= prediction["p90"]).all()
    assert all(0 <= v <= 1 for k, v in result.report.to_metrics().items() if "brier" in k and "minus" not in k)


def test_tuning_never_reads_the_test_split(cleaned: pd.DataFrame, config: dict) -> None:
    splits = make_splits(cleaned, config)
    corrupted = Splits(splits.train, splits.val, test=splits.test.iloc[0:0])

    assert tune_models(splits, config, enabled=True) == tune_models(corrupted, config, enabled=True)
