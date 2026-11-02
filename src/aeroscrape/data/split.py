"""Chronological train/validation/test split."""

import logging
from dataclasses import dataclass

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SplitSummary:
    rows: int
    start: pd.Timestamp
    end: pd.Timestamp


@dataclass(frozen=True)
class Splits:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame

    @property
    def train_val(self) -> pd.DataFrame:
        """Train and validation together, still in time order, for rolling-origin CV and the final fit."""
        return pd.concat([self.train, self.val])

    def summary(self, time_column: str) -> dict[str, SplitSummary]:
        return {
            name: SplitSummary(len(part), pd.Timestamp(part[time_column].min()), pd.Timestamp(part[time_column].max()))
            for name, part in (("train", self.train), ("val", self.val), ("test", self.test))
        }

    def describe(self, time_column: str) -> pd.DataFrame:
        return pd.DataFrame({name: vars(s) for name, s in self.summary(time_column).items()}).T


def time_split(df: pd.DataFrame, time_column: str, train_end: str, val_end: str) -> Splits:
    """Split rows by time: train < train_end <= val < val_end <= test.

    A random split on time-ordered data lets the model learn from flights that
    happen after the ones it is tested on, which overstates how well it works.
    """
    train_cut, val_cut = pd.Timestamp(train_end), pd.Timestamp(val_end)
    if train_cut >= val_cut:
        raise ValueError(f"train_end ({train_end}) must be before val_end ({val_end})")

    times = pd.to_datetime(df[time_column])
    if times.isna().any():
        raise ValueError(f"{int(times.isna().sum())} rows have no {time_column}")

    ordered = df.assign(**{time_column: times}).sort_values(time_column, kind="stable")
    splits = Splits(
        train=ordered[ordered[time_column] < train_cut],
        val=ordered[(ordered[time_column] >= train_cut) & (ordered[time_column] < val_cut)],
        test=ordered[ordered[time_column] >= val_cut],
    )
    empty = [name for name in ("train", "val", "test") if getattr(splits, name).empty]
    if empty:
        raise ValueError(f"Empty split(s) {empty}; check evaluation.train_end and evaluation.val_end")

    logger.info("Time split:\n%s", splits.describe(time_column).to_string())
    return splits
