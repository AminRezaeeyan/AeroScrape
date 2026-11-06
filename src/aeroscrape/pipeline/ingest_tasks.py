"""Airflow task callables for the ingest DAG: scrape the FIDS pages, validate, load into Postgres."""

import logging
import re
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd

from aeroscrape.config import get_app_config, resolve_path
from aeroscrape.data.clean import normalize_persian
from aeroscrape.data.validate import validate_raw_data
from aeroscrape.monitoring.push import GaugeValue, push_gauges
from aeroscrape.scraping.scraper import get_all_flights_data
from aeroscrape.storage.database import Database

logger = logging.getLogger(__name__)


def _run_dir(kwargs) -> Path:
    safe_run_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", kwargs["run_id"])
    path = resolve_path(get_app_config()["data_paths"]["runs_dir"]) / f"ingest_{safe_run_id}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def task_scrape(**kwargs):
    """Fetch every configured airport and save the raw rows; an empty scrape fails the run."""
    scraped_at = datetime.now()
    flights = get_all_flights_data()
    if flights is None or flights.empty:
        raise RuntimeError("The scrape returned no flights; the site may be down or its layout changed")
    path = _run_dir(kwargs) / "scraped.csv"
    flights.to_csv(path, index=False, encoding="utf-8-sig")
    logger.info("Scraped %d flights from %d airports", len(flights), flights["airport"].nunique())
    kwargs["ti"].xcom_push(key="scraped_path", value=str(path))
    kwargs["ti"].xcom_push(key="scraped_at", value=scraped_at.isoformat())


def scrape_metrics(flights: pd.DataFrame, airports: list[str]) -> list[GaugeValue]:
    """Rows per configured airport, zero for airports that returned nothing."""
    counts = flights["airport"].map(normalize_persian).value_counts()
    return [
        GaugeValue(
            "scraped_rows",
            "Rows scraped from an airport's board in the last run",
            int(counts.get(name, 0)),
            {"airport": name},
        )
        for name in (str(normalize_persian(a)) for a in airports)
    ]


def task_validate_scrape(**kwargs):
    """Validate the scraped rows; the scrape's metrics are pushed whether or not they pass."""
    ti = kwargs["ti"]
    path = ti.xcom_pull(task_ids="scrape", key="scraped_path")
    config = get_app_config()
    flights = pd.read_csv(path, dtype=str, encoding="utf-8-sig")
    values = scrape_metrics(flights, [a["name"] for a in config["scraper"]["airports"]])
    failed = GaugeValue("ingest_validation_failed", "1 if the last scrape failed validation", 1)
    try:
        validate_raw_data(path)
    except Exception:
        push_gauges("ingest_scrape", [*values, failed], config)
        raise
    push_gauges("ingest_scrape", [*values, GaugeValue(failed.name, failed.help, 0)], config)


def task_load(**kwargs):
    """Upsert into Postgres. Rejected rows or an empty load fail the task instead of passing silently."""
    ti = kwargs["ti"]
    flights = pd.read_csv(ti.xcom_pull(task_ids="scrape", key="scraped_path"), dtype=str, encoding="utf-8-sig")
    # Scheduled times are weekday + HH:MM, resolved against the moment of the scrape
    scraped_at = datetime.fromisoformat(ti.xcom_pull(task_ids="scrape", key="scraped_at"))
    result = Database().insert_flights_from_dataframe(flights.where(flights.notna(), None), base_date=scraped_at)
    push_gauges(
        "ingest_load",
        [
            GaugeValue("ingest_rows_loaded", "Flights upserted in the last load", result.inserted),
            GaugeValue("ingest_rows_rejected", "Flights the database rejected in the last load", result.failed),
            GaugeValue(
                "ingest_rows_skipped", "Flights skipped as incomplete in the last load", sum(result.skipped.values())
            ),
        ],
    )
    if result.failed or result.inserted == 0:
        raise RuntimeError(f"Flight load failed: {result.summary()}")
    # A job of its own, so a failed load doesn't erase when the last good one was
    push_gauges(
        "ingest_success",
        [GaugeValue("ingest_last_success_timestamp_seconds", "When a load last succeeded", datetime.now().timestamp())],
    )
    logger.info("Flight load: %s", result.summary())


def task_cleanup(**kwargs):
    shutil.rmtree(_run_dir(kwargs), ignore_errors=True)
