"""Where training data comes from: the scraped archive file and the flight database."""

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from aeroscrape.config import resolve_path
from aeroscrape.data.clean import CleaningAudit, clean_flights
from aeroscrape.storage.database import Database

logger = logging.getLogger(__name__)


def load_raw_flights(config: dict[str, Any], include_database: bool | None = None) -> pd.DataFrame:
    """Raw scraped flights from the archive CSV and, if enabled, everything in Postgres.

    Both are in the scraper's CSV format, so they are simply stacked; the cleaner removes
    flights that appear in both.
    """
    sources = config["data_sources"]
    parts = []
    archive = sources.get("archive")
    if archive:
        path = resolve_path(archive)
        parts.append(pd.read_csv(path, dtype=str, encoding="utf-8-sig"))
        logger.info("Archive %s: %d rows", path, len(parts[-1]))
    if sources.get("database", False) if include_database is None else include_database:
        db_rows = Database().fetch_flights().astype(str).where(lambda df: df != "None")
        parts.append(db_rows)
        logger.info("Flight database: %d rows", len(db_rows))
    if not parts:
        raise ValueError("No data source enabled in config.yaml under data_sources")
    return pd.concat(parts, ignore_index=True)


def build_dataset(config: dict[str, Any], output: str | Path, include_database: bool | None = None) -> CleaningAudit:
    """Clean every flight from the configured sources into one file for a training run."""
    audit = CleaningAudit()
    flights = clean_flights(load_raw_flights(config, include_database), audit)
    flights.to_csv(output, index=False, encoding="utf-8-sig")
    logger.info("Wrote %d cleaned flights to %s", len(flights), output)
    return audit
