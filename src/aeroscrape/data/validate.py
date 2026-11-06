import logging
from typing import Any

import pandas as pd
import pandera.pandas as pa
from pandera.errors import SchemaError

from aeroscrape.data.clean import normalize_persian

logger = logging.getLogger(__name__)

# Batch-level rules are strict: without these the rows are unusable or the page layout changed.
# A single flight with an empty status or airline is normal on the boards; the loader skips such
# rows with a counted reason. Only when many rows lack them does the whole batch fail.
MAX_INCOMPLETE_SHARE = 0.05
OPTIONAL_FIELDS = ["airline", "destination_or_origin", "status"]

raw_data_schema = pa.DataFrameSchema(
    {
        "scheduled_time": pa.Column(pa.String, nullable=False),
        "actual_time": pa.Column(pa.String, nullable=True),
        "airline": pa.Column(pa.String, nullable=True),
        "flight_number": pa.Column(pa.String, nullable=False),
        "destination_or_origin": pa.Column(pa.String, nullable=True),
        "status": pa.Column(pa.String, nullable=True),
        "flight_type": pa.Column(
            pa.String,
            pa.Check.isin(
                ["Domestic Departure", "Domestic Arrival", "International Arrival", "International Departure"]
            ),
        ),
    },
    checks=[
        pa.Check(
            lambda df: bool(df[OPTIONAL_FIELDS].isna().any(axis=1).mean() <= MAX_INCOMPLETE_SHARE),
            error=f"more than {MAX_INCOMPLETE_SHARE:.0%} of rows lack airline, destination or status",
        )
    ],
    strict=False,
)


def validate_raw_data(data_path):
    logger.info(f"Validating raw data from {data_path}")
    try:
        df = pd.read_csv(data_path)
        raw_data_schema.validate(df, lazy=True)
        logger.info("Raw data validation successful.")
        return True
    except FileNotFoundError:
        logger.error(f"Validation failed: Raw data file not found at {data_path}")
        raise
    except SchemaError as err:
        logger.error("Raw data validation failed!")
        logger.error(err.failure_cases)
        raise err
    except Exception as e:
        logger.error(f"An unexpected error occurred during validation: {e}")
        raise


class BatchTooSmallError(ValueError):
    """A scrape returned fewer rows than a normal run, overall or for one of the busiest airports."""


def check_row_counts(df: pd.DataFrame, min_rows: dict[str, Any]) -> list[str]:
    """Problems with the size of a scraped batch (empty if none)."""
    problems = []
    if len(df) < min_rows.get("total", 0):
        problems.append(f"{len(df)} rows in total, fewer than {min_rows['total']}")
    counts = df["airport"].map(normalize_persian).value_counts() if len(df) else pd.Series(dtype=int)
    for airport, minimum in (min_rows.get("per_airport") or {}).items():
        n = int(counts.get(normalize_persian(airport), 0))
        if n < minimum:
            problems.append(f"{airport}: {n} rows, fewer than {minimum}")
    return problems


def quality_metrics(df: pd.DataFrame) -> dict[str, float]:
    """Shares that drift when the site or the parser changes: empty cells and flights with an actual time."""
    if df.empty:
        return {}
    blank = df.replace("", None).isna()
    metrics = {
        f"null_share.{col}": float(blank[col].mean())
        for col in ("airline", "status", "register", "aircraft")
        if col in df
    }
    if "actual_time" in df:
        metrics["actual_time_share"] = float(1 - blank["actual_time"].mean())
    return metrics
