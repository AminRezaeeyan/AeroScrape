"""Airflow task callables for the ingest DAG: scrape the FIDS pages, validate, load into Postgres."""

import logging
import re
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd

from aeroscrape.config import get_app_config, resolve_path
from aeroscrape.data.validate import validate_raw_data
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


def task_validate_scrape(**kwargs):
    validate_raw_data(kwargs["ti"].xcom_pull(task_ids="scrape", key="scraped_path"))


def task_load(**kwargs):
    """Upsert into Postgres. Rejected rows or an empty load fail the task instead of passing silently."""
    ti = kwargs["ti"]
    flights = pd.read_csv(ti.xcom_pull(task_ids="scrape", key="scraped_path"), dtype=str, encoding="utf-8-sig")
    # Scheduled times are weekday + HH:MM, resolved against the moment of the scrape
    scraped_at = datetime.fromisoformat(ti.xcom_pull(task_ids="scrape", key="scraped_at"))
    result = Database().insert_flights_from_dataframe(flights.where(flights.notna(), None), base_date=scraped_at)
    if result.failed or result.inserted == 0:
        raise RuntimeError(f"Flight load failed: {result.summary()}")
    logger.info("Flight load: %s", result.summary())


def task_cleanup(**kwargs):
    shutil.rmtree(_run_dir(kwargs), ignore_errors=True)
