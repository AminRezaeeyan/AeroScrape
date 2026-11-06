"""The prediction log and monitoring tables in a real Postgres (AEROSCRAPE_TEST_DB=1, as in CI)."""

import os
from datetime import UTC, datetime, timedelta

import pandas as pd
import psycopg2
import pytest

from aeroscrape.monitoring.prediction_log import PredictionLog, build_records
from aeroscrape.monitoring.store import MonitoringStore
from aeroscrape.storage.database import Database

pytestmark = pytest.mark.skipif(os.getenv("AEROSCRAPE_TEST_DB") != "1", reason="needs a disposable Postgres")

FLIGHT = {
    "airline": "ماهان",
    "airport": "فرودگاه تست",
    "destination_or_origin": "مشهد",
    "scheduled_datetime": "2026-10-20 08:00:00",
}
OUT = pd.DataFrame({"p_delay_over_15": [0.7], "likely_delayed": [True], "p10": [5.0], "p50": [25.0], "p90": [70.0]})


def _log(source: str, flight_number: str | None) -> None:
    X = pd.DataFrame({"airline": ["ماهان"], "has_inbound": [1.0]})
    flight = FLIGHT | ({"flight_number": flight_number} if flight_number else {})
    PredictionLog(Database()).write(build_records("r", source, [flight], X, OUT, "7"))


def test_logged_predictions_are_matched_and_scored_once() -> None:
    db = Database()
    store = MonitoringStore(db)
    _log("replay", "TST900")
    _log("loadtest", "TST901")  # load tests never reach monitoring
    _log("client", None)  # no flight number: drift only

    window = datetime(2026, 10, 20), datetime(2026, 10, 21)
    pending = store.unmatched_predictions(*window)
    assert pending["flight_key"].tolist() == ["فرودگاه تست|TST900|2026-10-20 08:00"]

    outcomes = pd.DataFrame({"prediction_id": pending["id"], "actual_delay_min": [32.0]})
    assert store.insert_outcomes(outcomes) == 1
    assert store.insert_outcomes(outcomes) == 0  # matching again changes nothing
    assert store.unmatched_predictions(*window).empty

    scored = store.scored_flights(*window)
    assert scored[["actual_delay_min", "has_inbound"]].to_dict("records") == [
        {"actual_delay_min": 32.0, "has_inbound": "1.0"}
    ]
    inputs = store.recent_inputs(datetime.now(UTC) - timedelta(minutes=5))
    assert len(inputs) == 2 and set(inputs.columns) == {"airline", "has_inbound"}

    with db.get_connection() as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM predictions WHERE airport = 'فرودگاه تست'")


@pytest.mark.skipif(not os.getenv("AEROSCRAPE_TEST_READER"), reason="needs the API role from the init scripts")
def test_the_api_role_can_log_predictions_but_not_read_them() -> None:
    user = os.environ["AEROSCRAPE_TEST_READER"]
    params = Database().conn_params | {"user": user, "password": user}
    conn = psycopg2.connect(**params)
    try:
        with conn, conn.cursor() as cur:
            cur.execute(
                "INSERT INTO predictions (request_id, airport, airline, scheduled_time, model_version, p_delay,"
                " likely_delayed, q10, q50, q90, features) VALUES ('r', 'a', 'b', now(), '7', 0.5, false, 0, 1, 2, '{}')"
            )
        with pytest.raises(psycopg2.errors.InsufficientPrivilege), conn, conn.cursor() as cur:
            cur.execute("SELECT * FROM predictions")
    finally:
        conn.close()
    with Database().get_connection() as admin, admin.cursor() as cur:
        cur.execute("DELETE FROM predictions WHERE request_id = 'r' AND airport = 'a'")
