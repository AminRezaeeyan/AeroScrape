"""Airflow task callables for the ingest DAG: scrape the FIDS pages, validate, load into Postgres.

A page that couldn't be fetched or parsed doesn't stop the others from loading, but the
run still fails at the end (check_pages), so it is visible in Airflow and in Prometheus.
"""

import logging
import re
import shutil
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from aeroscrape.config import get_app_config, local_now, resolve_path
from aeroscrape.data.clean import normalize_persian
from aeroscrape.data.validate import BatchTooSmallError, check_row_counts, quality_metrics, validate_raw_data
from aeroscrape.monitoring.push import GaugeValue, push_gauges
from aeroscrape.scraping.scraper import PageResult, PageStatus, scrape
from aeroscrape.storage.database import Database

logger = logging.getLogger(__name__)


def _run_dir(kwargs) -> Path:
    safe_run_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", kwargs["run_id"])
    path = resolve_path(get_app_config()["data_paths"]["runs_dir"]) / f"ingest_{safe_run_id}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def scrape_metrics(pages: list[PageResult]) -> list[GaugeValue]:
    """Rows and failure per configured airport; an airport that failed has 0 rows."""
    values = []
    for page in pages:
        labels = {"airport": str(normalize_persian(page.airport))}
        values.append(
            GaugeValue("scraped_rows", "Rows scraped from an airport's board in the last run", page.rows, labels)
        )
        failed = int(page.status != PageStatus.OK)
        values.append(
            GaugeValue("scrape_page_failed", "1 if the airport's page couldn't be fetched or parsed", failed, labels)
        )
    return values


def task_scrape(**kwargs):
    """Fetch every airport, keep the raw pages, and save the parsed rows; no rows at all fails the run."""
    config = get_app_config()
    scraped_at = local_now(config)  # weekday names on the boards are resolved against this
    raw_dir = (
        resolve_path(config["scraper"]["raw_html_dir"])
        / scraped_at.strftime("%Y-%m-%d")
        / scraped_at.strftime("%H%M%S")
    )
    result = scrape(raw_dir, config)
    push_gauges("ingest_scrape", scrape_metrics(result.pages), config)
    for page in result.failures:
        logger.error("%s: %s (%s)", page.airport, page.status, page.error)
    if result.flights.empty:
        raise RuntimeError(f"The scrape returned no flights ({len(result.failures)} pages failed); see {raw_dir}")

    path = _run_dir(kwargs) / "scraped.csv"
    result.flights.to_csv(path, index=False, encoding="utf-8-sig")
    logger.info(
        "Scraped %d flights from %d airports; %d pages failed; raw pages in %s",
        len(result.flights),
        sum(p.rows > 0 for p in result.pages),
        len(result.failures),
        raw_dir,
    )
    ti = kwargs["ti"]
    ti.xcom_push(key="scraped_path", value=str(path))
    ti.xcom_push(key="scraped_at", value=scraped_at.isoformat())
    ti.xcom_push(key="failed_pages", value=[asdict(p) for p in result.failures])


def task_validate_scrape(**kwargs):
    """The data contract and the batch size; data-quality numbers are pushed either way."""
    config = get_app_config()
    path = kwargs["ti"].xcom_pull(task_ids="scrape", key="scraped_path")
    flights = pd.read_csv(path, dtype=str, encoding="utf-8-sig")
    quality = [
        GaugeValue(f"scrape_{name.replace('.', '_')}", "Data quality of the last scrape", value)
        for name, value in quality_metrics(flights).items()
    ]

    def push(failed: int) -> None:
        verdict = GaugeValue("ingest_validation_failed", "1 if the last scrape failed validation", failed)
        push_gauges("ingest_validate", [*quality, verdict], config)

    try:
        validate_raw_data(path)
        problems = check_row_counts(flights, config["scraper"].get("min_rows", {}))
        if problems:
            raise BatchTooSmallError("; ".join(problems))
    except Exception:
        push(1)
        raise
    push(0)


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
            GaugeValue("ingest_rows_loaded", "Flights new or changed in the last load", result.inserted),
            GaugeValue("ingest_rows_unchanged", "Flights already stored as scraped", result.unchanged),
            GaugeValue("ingest_rows_rejected", "Flights the database rejected in the last load", result.failed),
            GaugeValue(
                "ingest_rows_skipped", "Flights skipped as incomplete in the last load", sum(result.skipped.values())
            ),
        ],
    )
    if result.failed or result.inserted + result.unchanged == 0:
        raise RuntimeError(f"Flight load failed: {result.summary()}")
    # A job of its own, so a failed load doesn't erase when the last good one was
    push_gauges(
        "ingest_success",
        [GaugeValue("ingest_last_success_timestamp_seconds", "When a load last succeeded", datetime.now().timestamp())],
    )
    logger.info("Flight load: %s", result.summary())


def failed_pages_message(failed: list[dict[str, Any]]) -> str | None:
    if not failed:
        return None
    return f"{len(failed)} airport pages failed: " + "; ".join(
        f"{p['airport']}: {p['status']} ({p['error']})" for p in failed
    )


def task_check_pages(**kwargs):
    """Fails the run when a page couldn't be fetched or parsed, after the other airports were loaded."""
    from airflow.exceptions import AirflowFailException

    message = failed_pages_message(kwargs["ti"].xcom_pull(task_ids="scrape", key="failed_pages") or [])
    if message:
        raise AirflowFailException(message)  # no retries: the scrape itself would have to run again


def remove_old_pages(raw_root: Path, keep_days: int, today: datetime | None = None) -> list[Path]:
    """Delete date folders of saved pages older than `keep_days`; returns what was removed."""
    cutoff = (today or datetime.now()).date() - timedelta(days=keep_days)
    removed = []
    for folder in sorted(raw_root.glob("????-??-??")):
        try:
            day = datetime.strptime(folder.name, "%Y-%m-%d").date()
        except ValueError:
            continue
        if day < cutoff:
            shutil.rmtree(folder, ignore_errors=True)
            removed.append(folder)
    return removed


def task_cleanup(**kwargs):
    shutil.rmtree(_run_dir(kwargs), ignore_errors=True)
    scraper_cfg = get_app_config()["scraper"]
    removed = remove_old_pages(
        resolve_path(scraper_cfg["raw_html_dir"]), scraper_cfg["raw_html_retention_days"], local_now()
    )
    if removed:
        logger.info("Removed saved pages of %d old days", len(removed))
