from pathlib import Path

from aeroscrape.data.clean import clean_flights_file, read_flights, select_training_rows
from aeroscrape.data.validate import validate_raw_data


def test_sample_passes_raw_schema(sample_raw_path: Path) -> None:
    assert validate_raw_data(sample_raw_path) is True


def test_cleaning_the_sample_keeps_all_flight_types(sample_raw_path: Path, tmp_path: Path) -> None:
    output = tmp_path / "cleaned.csv"
    audit = clean_flights_file(sample_raw_path, output)
    flights = read_flights(output)

    assert len(flights) == 200
    assert set(flights["direction"]) == {"arrival", "departure"}
    rows = select_training_rows(flights)
    assert len(rows) == dict(audit.steps)[audit.steps[-1][0]]
    assert rows["delay_minutes"].notna().all()
