"""Model inputs: training rows with features and targets, feature sets, and the preprocessor."""

import logging
from dataclasses import dataclass
from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from aeroscrape.data.clean import CleaningAudit, select_training_rows
from aeroscrape.features.build import FeatureSettings, build_features

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FeatureSet:
    numerical: tuple[str, ...]
    categorical: tuple[str, ...]

    @property
    def columns(self) -> list[str]:
        return [*self.numerical, *self.categorical]


def feature_set(config: dict[str, Any], groups: list[str]) -> FeatureSet:
    defined = config["features"]["groups"]
    unknown = [g for g in groups if g not in defined]
    if unknown:
        raise KeyError(f"Unknown feature group(s) {unknown}; defined: {list(defined)}")
    numerical = [c for g in groups for c in defined[g].get("numerical", [])]
    categorical = [c for g in groups for c in defined[g].get("categorical", [])]
    return FeatureSet(tuple(dict.fromkeys(numerical)), tuple(dict.fromkeys(categorical)))


def variant_feature_set(config: dict[str, Any], variant: str) -> FeatureSet:
    return feature_set(config, config["features"]["variants"][variant])


def ablation_steps(config: dict[str, Any]) -> list[tuple[str, FeatureSet]]:
    """Cumulative feature sets: the first step, then each later group added on top."""
    steps, groups = [], []
    for i, step in enumerate(config["features"]["ablation"]):
        groups += step
        label = " + ".join(step) if i == 0 else f"+ {' + '.join(step)}"
        steps.append((label, feature_set(config, list(groups))))
    return steps


def make_model_frame(flights: pd.DataFrame, config: dict[str, Any], audit: CleaningAudit | None = None) -> pd.DataFrame:
    """Training rows (departed domestic departures) with every feature and both targets."""
    pipeline_cfg = config["pipeline"]
    rows = select_training_rows(flights, audit)
    frame = build_features(rows, flights, FeatureSettings.from_config(config))
    frame[pipeline_cfg["target_classification"]] = (
        frame[pipeline_cfg["target_regression"]] > pipeline_cfg["delay_threshold_minutes"]
    ).astype(int)
    return frame


def prepare_input(X: pd.DataFrame, features: FeatureSet) -> pd.DataFrame:
    """The model's input columns with stable types: floats for numbers, strings for categories.

    Training and serving both go through this, so a value arriving as JSON gets the same
    type the model was fit on.
    """
    out = X.reindex(columns=features.columns)
    for col in features.numerical:
        out[col] = pd.to_numeric(out[col], errors="coerce").astype(float)
    for col in features.categorical:
        out[col] = out[col].astype(str)
    return out


def feature_coverage(frame: pd.DataFrame, features: FeatureSet) -> pd.Series:
    """Share of rows where each numerical feature is known."""
    return frame[list(features.numerical)].notna().mean()


def _as_text(X: pd.DataFrame) -> pd.DataFrame:
    return X.astype(str)


def build_preprocessor(features: FeatureSet) -> ColumnTransformer:
    numerical = Pipeline(
        [
            # Missing usually means "no linked flight", which is informative, hence the indicator
            ("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
            ("scale", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            # Categories are compared as strings, so 1 and "1" (or a NaN read back from JSON) agree
            ("as_text", FunctionTransformer(_as_text, feature_names_out="one-to-one")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numerical, list(features.numerical)),
            ("cat", categorical, list(features.categorical)),
        ],
        remainder="drop",
    )


def load_data(data_path):
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    logger.info(f"Data loaded successfully. Shape: {df.shape}")
    return df
