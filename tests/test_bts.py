import copy
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aeroscrape.config import apply_profile, get_app_config
from aeroscrape.scraping import bts
from aeroscrape.scraping.bts import download_month, load_bts_flights, month_path, read_month, to_flights
from tests.bts_synthetic import make_bts_frame, write_bts_zip


def _row(**overrides) -> dict:
    row = {
        "FlightDate": "2026-01-10",
        "Reporting_Airline": "DL",
        "Tail_Number": "N123DL",
        "Flight_Number_Reporting_Airline": 1234,
        "Origin": "ATL",
        "Dest": "BOS",
        "CRSDepTime": 830,
        "DepDelay": 12.0,
        "CRSArrTime": 1105,
        "ArrDelay": 5.0,
        "Cancelled": 0.0,
        "Diverted": 0.0,
    }
    return row | overrides


def test_one_flight_becomes_a_departure_and_an_arrival() -> None:
    flights = to_flights(pd.DataFrame([_row()]))
    dep = flights[flights["direction"] == "departure"].iloc[0]
    arr = flights[flights["direction"] == "arrival"].iloc[0]

    assert (dep["airport"], dep["destination_or_origin"], dep["flight_number"]) == ("ATL", "BOS", "DL1234")
    assert dep["scheduled_datetime"] == pd.Timestamp("2026-01-10 08:30")
    assert dep["actual_datetime"] == pd.Timestamp("2026-01-10 08:42") and dep["delay_minutes"] == 12
    assert dep["status"] == "departed" and dep["register"] == "N123DL"
    assert (arr["airport"], arr["destination_or_origin"], arr["status"]) == ("BOS", "ATL", "landed")
    assert arr["scheduled_datetime"] == pd.Timestamp("2026-01-10 11:05") and arr["delay_minutes"] == 5


def test_midnight_and_overnight_arrival() -> None:
    flights = to_flights(pd.DataFrame([_row(CRSDepTime=2400, CRSArrTime=230)]))
    dep = flights[flights["direction"] == "departure"].iloc[0]
    arr = flights[flights["direction"] == "arrival"].iloc[0]
    assert dep["scheduled_datetime"] == pd.Timestamp("2026-01-11 00:00")
    assert arr["scheduled_datetime"] == pd.Timestamp("2026-01-11 02:30")

    red_eye = to_flights(pd.DataFrame([_row(CRSDepTime=2330, CRSArrTime=545)]))
    assert red_eye[red_eye["direction"] == "arrival"].iloc[0]["scheduled_datetime"] == pd.Timestamp("2026-01-11 05:45")


def test_cancelled_and_diverted_flights_have_no_delay() -> None:
    flights = to_flights(
        pd.DataFrame(
            [
                _row(Cancelled=1.0, DepDelay=np.nan, ArrDelay=np.nan),
                _row(Flight_Number_Reporting_Airline=99, Diverted=1.0, ArrDelay=np.nan),
            ]
        )
    )
    assert set(flights.loc[flights["flight_number"] == "DL1234", "status"]) == {"cancelled"}
    assert flights.loc[flights["flight_number"] == "DL99", "status"].eq("unknown").all()
    assert flights["delay_minutes"].isna().all()


def test_bad_tail_numbers_are_dropped_and_extreme_delays_ignored() -> None:
    flights = to_flights(pd.DataFrame([_row(Tail_Number="-"), _row(Flight_Number_Reporting_Airline=7, DepDelay=900.0)]))
    assert flights.loc[flights["flight_number"] == "DL1234", "register"].isna().all()
    assert flights.loc[flights["flight_number"] == "DL7", "register"].eq("N123DL").all()
    departure = flights[(flights["flight_number"] == "DL7") & (flights["direction"] == "departure")].iloc[0]
    assert pd.isna(departure["delay_minutes"])  # 900 min is outside the plausible range


def test_read_month_keeps_flights_touching_the_airports(tmp_path: Path) -> None:
    frame = pd.DataFrame([_row(), _row(Origin="ORD", Dest="DEN"), _row(Origin="MIA", Dest="ATL")])
    path = write_bts_zip(tmp_path / "m.zip", frame)
    assert len(read_month(path, ["ATL"])) == 2
    assert len(read_month(path, None)) == 3


def test_download_uses_the_cache_and_rejects_non_zip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cached = write_bts_zip(month_path(tmp_path, 2026, 1), pd.DataFrame([_row()]))
    monkeypatch.setattr(bts.requests, "get", lambda *a, **k: pytest.fail("should not download a cached month"))
    assert download_month(2026, 1, tmp_path) == cached

    class Html:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def raise_for_status(self):
            pass

        def iter_content(self, chunk_size):
            yield b"<html>not published</html>"

    monkeypatch.setattr(bts.requests, "get", lambda *a, **k: Html())
    with pytest.raises(RuntimeError, match="did not return a zip"):
        download_month(2026, 2, tmp_path)
    assert not any(p.suffix == ".part" for p in tmp_path.iterdir())


@pytest.fixture(scope="module")
def synthetic_cache(tmp_path_factory) -> Path:
    cache = tmp_path_factory.mktemp("bts")
    write_bts_zip(month_path(cache, 2026, 1), make_bts_frame(date(2026, 1, 1), days=28))
    return cache


def test_load_from_cache_and_run_the_pipeline_on_the_bts_profile(synthetic_cache: Path) -> None:
    from aeroscrape.models.experiment import run_experiment

    config = copy.deepcopy(apply_profile(get_app_config(), "bts"))
    flights, files = load_bts_flights(["2026-01"], ["ATL"], synthetic_cache, download=False)
    assert files == [month_path(synthetic_cache, 2026, 1)]

    config["evaluation"] |= {"test_dates": 5, "val_dates": 5, "bootstrap_samples": 50, "cv_splits": 2}
    for key in ("lgbm_params", "lgbm_classifier_params"):
        config["pipeline"][key] = config["pipeline"][key] | {"n_estimators": 30}
    result = run_experiment(flights, config, tune=False, raw_path=files)

    report = result.report.to_markdown()
    assert "LightGBM median (1 h before departure)" in report
    assert "BTS records the tail number" in report
    # Rotation features exist for most departures in BTS-like data
    assert "| rotation | `nth_flight_of_day_for_aircraft` |" in report
    assert result.served.metadata["lineage"]["raw_data_sha256"] != "unknown"


def test_learning_curve_on_synthetic_bts(tmp_path: Path) -> None:
    from aeroscrape.features.preprocess import make_model_frame
    from aeroscrape.models.learning_curve import curve_figure, curve_markdown, learning_curve

    write_bts_zip(month_path(tmp_path, 2026, 1), make_bts_frame(date(2026, 1, 1), days=70, tails=12))
    config = copy.deepcopy(apply_profile(get_app_config(), "bts"))
    config["evaluation"] |= {"bootstrap_samples": 50}
    config["pipeline"]["lgbm_params"] = config["pipeline"]["lgbm_params"] | {"n_estimators": 30}
    flights, _ = load_bts_flights(["2026-01"], ["ATL"], tmp_path, download=False)

    points = learning_curve(make_model_frame(flights, config), config, [1, 2, 4, 8, 16], "pre_departure")

    assert [p.weeks for p in points] == [1, 2, 4, 8]  # the data doesn't reach back 16 weeks
    assert [p.train_rows for p in points] == sorted(p.train_rows for p in points)
    assert "| 8 |" in curve_markdown(points, "bts")
    curve_figure(points, "test", reference_rows=500).savefig(tmp_path / "curve.png")
