import pandas as pd
import pytest

from aeroscrape.data.split import time_split


@pytest.fixture
def flights() -> pd.DataFrame:
    times = pd.date_range("2025-06-01", "2025-06-12 23:00", freq="6h")
    # Shuffled on purpose: the split must not depend on input order
    return pd.DataFrame({"scheduled_datetime": times, "delay": range(len(times))}).sample(frac=1, random_state=0)


def test_split_is_chronological_and_disjoint(flights: pd.DataFrame) -> None:
    splits = time_split(flights, "scheduled_datetime", "2025-06-06", "2025-06-10")

    assert splits.train.scheduled_datetime.max() < pd.Timestamp("2025-06-06")
    assert splits.val.scheduled_datetime.min() >= pd.Timestamp("2025-06-06")
    assert splits.val.scheduled_datetime.max() < pd.Timestamp("2025-06-10")
    assert splits.test.scheduled_datetime.min() >= pd.Timestamp("2025-06-10")
    assert len(splits.train) + len(splits.val) + len(splits.test) == len(flights)
    assert splits.train_val.scheduled_datetime.is_monotonic_increasing


def test_empty_split_is_an_error(flights: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="Empty split"):
        time_split(flights, "scheduled_datetime", "2025-06-06", "2030-01-01")


def test_cut_dates_must_be_ordered(flights: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="must be before"):
        time_split(flights, "scheduled_datetime", "2025-06-10", "2025-06-06")
