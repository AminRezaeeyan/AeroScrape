import json

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from prometheus_client import generate_latest

from aeroscrape.monitoring.drift import detect_drift, drift_html, feature_drift
from aeroscrape.monitoring.performance import (
    actual_delays,
    daily_performance,
    degraded_days,
    match_outcomes,
    offline_metrics,
    summary_for,
)
from aeroscrape.monitoring.prediction_log import build_records, flight_key
from aeroscrape.monitoring.push import GaugeValue, build_registry, push_gauges
from aeroscrape.pipeline.ingest_tasks import scrape_metrics
from aeroscrape.pipeline.monitor_tasks import retrain_reasons
from aeroscrape.serving import api
from tests.test_api import FLIGHT, KNOWN, FakeBundle

# ---------------------------------------------------------------------------- prediction log


def test_flight_key_matches_the_cleaner_spelling_and_needs_a_flight_number() -> None:
    # Arabic yeh and kaf and extra spaces normalize to the cleaner's form
    assert flight_key("فرودگاه  شيراز", " IR 452 ", "2026-11-06 08:05:00") == "فرودگاه شیراز|IR 452|2026-11-06 08:05"
    assert flight_key("فرودگاه مهرآباد", None, "2026-11-06 08:05") is None
    assert flight_key("فرودگاه مهرآباد", float("nan"), "2026-11-06 08:05") is None


def test_records_keep_the_model_input_as_json() -> None:
    X = pd.DataFrame(
        {"airline": ["ماهان"], "airport": ["فرودگاه مهرآباد"], "inbound_delay_min": [np.nan], "hour": [np.int64(8)]}
    )
    out = pd.DataFrame({"p_delay_over_15": [0.7], "likely_delayed": [True], "p10": [5.0], "p50": [25.0], "p90": [70.0]})
    flight = FLIGHT | {"flight_number": "W5 1012"}
    [record] = build_records("req-1", "replay", [flight], X, out, "7")

    assert record.flight_key == "فرودگاه مهرآباد|W5 1012|2025-06-01 08:00"
    assert (record.p_delay, record.q50, record.model_version, record.traffic_source) == (0.7, 25.0, "7", "replay")
    assert json.loads(record.features) == {
        "airline": "ماهان",
        "airport": "فرودگاه مهرآباد",
        "inbound_delay_min": None,
        "hour": 8,
    }
    assert len(record.as_row()) == 14


def test_api_logs_predictions_after_responding_and_exposes_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeBundle()
    monkeypatch.setattr(api, "served", api.ServedModel(fake, fake, "7", KNOWN))
    monkeypatch.setattr(api, "lookup", None)
    monkeypatch.setattr(api, "API_KEY", None)
    written = []

    class Log:
        def write(self, records):
            written.extend(records)

    monkeypatch.setattr(api, "prediction_log", Log())
    client = TestClient(api.app)
    other_airport = FLIGHT | {"airport": "فرودگاه ناشناخته"}
    response = client.post(
        "/v1/predict/batch", json={"flights": [FLIGHT, other_airport]}, headers={"X-Traffic-Source": "replay"}
    )

    assert response.status_code == 200
    assert [r.traffic_source for r in written] == ["replay", "replay"]
    assert written[0].request_id == response.headers["X-Request-ID"]
    metrics = client.get("/metrics").text
    assert (
        'predictions_total{airport="فرودگاه مهرآباد",endpoint="batch",likely_delayed="true",traffic_source="replay"}'
        in metrics
    )
    # An airport the model doesn't know doesn't become a label of its own
    assert 'airport="other"' in metrics and "ناشناخته" not in metrics
    assert "model_version_info" in metrics


def test_a_failing_prediction_log_does_not_fail_the_request(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeBundle()
    monkeypatch.setattr(api, "served", api.ServedModel(fake, fake, "7", KNOWN))
    monkeypatch.setattr(api, "lookup", None)
    monkeypatch.setattr(api, "API_KEY", None)

    class Broken:
        def write(self, records):
            raise ConnectionError("database is down")

    monkeypatch.setattr(api, "prediction_log", Broken())
    before = api.metrics.LOG_FAILURES._value.get()
    assert TestClient(api.app).post("/v1/predict", json=FLIGHT).status_code == 200
    assert api.metrics.LOG_FAILURES._value.get() == before + 1


def test_unknown_traffic_sources_count_as_clients() -> None:
    class Req:
        headers = {"X-Traffic-Source": "anything"}

    assert api._traffic_source(Req()) == "client"  # type: ignore[arg-type]


# ---------------------------------------------------------------------------- outcomes and performance


def _cleaned(rows: list[dict]) -> pd.DataFrame:
    base = {"flight_type": "Domestic Departure", "status": "departed", "airport": "فرودگاه مهرآباد"}
    df = pd.DataFrame([base | r for r in rows])
    df["scheduled_datetime"] = pd.to_datetime(df["scheduled_datetime"])
    df["actual_datetime"] = df["scheduled_datetime"] + pd.to_timedelta(df["delay_minutes"].fillna(0), unit="min")
    return df


def test_outcomes_use_the_training_labels() -> None:
    cleaned = _cleaned(
        [
            {"flight_number": "IR 1", "scheduled_datetime": "2026-11-06 08:00", "delay_minutes": 30.0},
            {
                "flight_number": "IR 2",
                "scheduled_datetime": "2026-11-06 09:00",
                "delay_minutes": 5.0,
                "status": "cancelled",
            },
            {
                "flight_number": "IR 3",
                "scheduled_datetime": "2026-11-06 10:00",
                "delay_minutes": 12.0,
                "flight_type": "Domestic Arrival",
            },
        ]
    )
    actual = actual_delays(cleaned)
    assert actual["flight_key"].tolist() == ["فرودگاه مهرآباد|IR 1|2026-11-06 08:00"]

    predictions = pd.DataFrame(
        {
            "id": [10, 11, 12],
            "flight_key": [actual["flight_key"][0], actual["flight_key"][0], "elsewhere|X|2026-11-06 08:00"],
        }
    )
    outcomes = match_outcomes(predictions, actual)
    assert outcomes.to_dict("list") == {"prediction_id": [10, 11], "actual_delay_min": [30.0, 30.0]}


def _scored(
    n: int, day: str, q50: float, actual: float | None, version: str = "5", created: str = "2026-11-01"
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "flight_key": [f"k{day}{i}" for i in range(n)],
            "model_version": version,
            "scheduled_time": pd.Timestamp(day),
            "created_at": pd.Timestamp(created),
            "p_delay": 0.8,
            "q10": 0.0,
            "q50": q50,
            "q90": 60.0,
            "has_inbound": "1",
            "actual_delay_min": actual,
        }
    )


def test_daily_performance_counts_each_flight_once_with_its_latest_prediction() -> None:
    first = _scored(40, "2026-11-05", q50=100.0, actual=20.0, created="2026-11-01")
    latest = _scored(40, "2026-11-05", q50=30.0, actual=20.0, created="2026-11-04")
    daily = daily_performance(pd.concat([first, latest]), min_outcomes=30)

    [row] = daily.to_dict("records")
    assert (row["predicted_flights"], row["matched_flights"]) == (40, 40)
    assert row["mae"] == pytest.approx(10.0)  # the later prediction, |30 - 20|
    assert row["brier"] == pytest.approx(0.04)  # p 0.8, every flight delayed over 15 min
    assert row["coverage"] == 1.0 and row["no_inbound_share"] == 0.0


def test_a_day_with_few_outcomes_has_no_metrics() -> None:
    rows = pd.concat(
        [
            _scored(10, "2026-11-05", 30.0, 20.0),
            _scored(20, "2026-11-05", 30.0, None).assign(flight_key=lambda d: d.flight_key + "u"),
        ]
    )
    [row] = daily_performance(rows, min_outcomes=30).to_dict("records")
    assert row["mae"] is None and row["matched_flights"] == 10
    assert row["no_outcome_share"] == pytest.approx(2 / 3)


def test_degraded_days_counts_the_recent_streak() -> None:
    daily = pd.DataFrame({"day": pd.date_range("2026-11-01", periods=5).date, "mae": [30.0, 19.0, 26.0, None, 27.0]})
    assert (
        degraded_days(daily, offline_mae=20.0, tolerance=0.25) == 2
    )  # 27 and 26; the day without enough outcomes is skipped
    assert degraded_days(daily, offline_mae=None, tolerance=0.25) == 0
    assert summary_for(daily.assign(model_version="5"), "5")["mae"] == 27.0


def test_offline_metrics_read_the_served_variants_test_numbers() -> None:
    run_metrics = {
        "test/lightgbm_median_at_booking/mae_min": 20.2,
        "test/lightgbm_calibrated_at_booking/brier": 0.225,
        "test/lightgbm_percentiles_conformal_at_booking/coverage_of_10_90_interval": 0.867,
        "test/lightgbm_median_1_h_before_departure/mae_min": 18.3,
    }
    assert offline_metrics(run_metrics, "booking") == {"mae": 20.2, "brier": 0.225, "coverage": 0.867}


# ---------------------------------------------------------------------------- drift


@pytest.fixture
def reference() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "Scheduled_Hour_of_Day": rng.integers(5, 24, 5000).astype(float),
            "scheduled_turnaround_min": rng.normal(60, 15, 5000),
            "has_inbound": rng.integers(0, 2, 5000).astype(float),
            "airline": rng.choice(["ماهان", "آتا", "کاسپین"], 5000),
        }
    )


NUMERICAL = {"Scheduled_Hour_of_Day", "scheduled_turnaround_min", "has_inbound"}


def test_a_sample_of_the_same_distribution_is_stable(reference: pd.DataFrame) -> None:
    current = reference.sample(1000, random_state=1)
    results = detect_drift(reference, current, list(reference.columns), NUMERICAL, threshold=0.25)
    assert all(r.psi < 0.1 for r in results), [(r.feature, r.psi) for r in results]
    assert not any(r.drifted for r in results)


def test_flights_moved_to_night_hours_drift(reference: pd.DataFrame) -> None:
    night = reference.sample(1000, random_state=1).assign(Scheduled_Hour_of_Day=lambda d: d.Scheduled_Hour_of_Day % 5)
    [hour] = detect_drift(reference, night, ["Scheduled_Hour_of_Day"], NUMERICAL, threshold=0.25)
    assert hour.drifted and hour.psi > 1
    assert hour.ks_pvalue is not None and hour.ks_pvalue < 1e-6


def test_a_feature_going_missing_is_drift(reference: pd.DataFrame) -> None:
    current = reference.sample(1000, random_state=1).assign(scheduled_turnaround_min=np.nan)
    result = feature_drift(reference["scheduled_turnaround_min"], current["scheduled_turnaround_min"], True, 0.25)
    assert result.drifted
    assert "(missing)" in result.shares["bin"].tolist()


def test_new_categories_and_flags_are_binned(reference: pd.DataFrame) -> None:
    current = reference.sample(1000, random_state=1).assign(airline="قشم ایر", has_inbound=0.0)
    airline, flag = detect_drift(reference, current, ["airline", "has_inbound"], NUMERICAL, threshold=0.25)
    assert airline.drifted and "(other)" in airline.shares["bin"].tolist()
    assert flag.drifted and set(flag.shares["bin"]) == {"0.0", "1.0"}  # a flag compares its values, not deciles


def test_drift_report_is_html_with_every_feature(reference: pd.DataFrame) -> None:
    results = detect_drift(reference, reference.head(500), list(reference.columns), NUMERICAL, 0.25)
    report = drift_html(results, "Input drift", 0.25)
    assert report.startswith("<!doctype html>")
    assert all(f"<h2>{r.feature} " in report for r in results)


# ---------------------------------------------------------------------------- pushing and retraining


def test_gauges_with_labels_build_one_metric_family() -> None:
    registry = build_registry(
        [
            GaugeValue("scraped_rows", "Rows", 12, {"airport": "a"}),
            GaugeValue("scraped_rows", "Rows", 0, {"airport": "b"}),
            GaugeValue("ingest_validation_failed", "Failed", 0),
        ]
    )
    text = generate_latest(registry).decode()
    assert 'scraped_rows{airport="a"} 12.0' in text and 'scraped_rows{airport="b"} 0.0' in text
    assert "ingest_validation_failed 0.0" in text


def test_pushing_never_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    values = [GaugeValue("x", "x", 1)]
    monkeypatch.setenv("PUSHGATEWAY_URL", "")
    assert push_gauges("job", values) is False  # disabled
    monkeypatch.setenv("PUSHGATEWAY_URL", "127.0.0.1:9")  # nothing listens there
    assert push_gauges("job", values) is False


def test_scrape_metrics_report_every_airport_and_its_failures() -> None:
    from aeroscrape.scraping.scraper import PageResult

    pages = [
        PageResult("فرودگاه مهرآباد", "u", "ok", rows=244),
        PageResult("فرودگاه شيراز", "u", "fetch_failed"),
        PageResult("فرودگاه زابل", "u", "ok", rows=0),
    ]
    values = {(v.name, v.labels["airport"]): v.value for v in scrape_metrics(pages)}
    assert values[("scraped_rows", "فرودگاه مهرآباد")] == 244
    assert values[("scrape_page_failed", "فرودگاه شیراز")] == 1  # normalized spelling
    assert values[("scrape_page_failed", "فرودگاه زابل")] == 0 and values[("scraped_rows", "فرودگاه زابل")] == 0


def test_retraining_needs_a_sustained_degradation_or_drift() -> None:
    cfg = {"degraded_days": 3}
    assert retrain_reasons(2, [], cfg) == []
    assert retrain_reasons(3, [], cfg) == ["live MAE degraded for 3 days"]
    assert retrain_reasons(0, ["Scheduled_Hour_of_Day"], cfg) == ["input drift in Scheduled_Hour_of_Day"]
