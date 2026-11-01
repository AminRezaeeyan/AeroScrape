from pathlib import Path

import pandas as pd

from aeroscrape.data.clean import clean_and_engineer_features
from aeroscrape.data.validate import validate_raw_data


def test_sample_passes_raw_schema(sample_raw_path: Path) -> None:
    assert validate_raw_data(sample_raw_path) is True


def test_clean_keeps_only_departed_domestic_flights(sample_raw_path: Path, tmp_path: Path) -> None:
    output = tmp_path / "cleaned.csv"
    clean_and_engineer_features(sample_raw_path, output)

    cleaned = pd.read_csv(output)
    assert len(cleaned) > 0
    assert set(cleaned["flight_type"]) == {"Domestic Departure"}
    assert cleaned["delay_minutes"].notna().all()
    assert {"scheduled_day_of_week", "is_weekend", "Scheduled_Hour_of_Day"} <= set(cleaned.columns)
