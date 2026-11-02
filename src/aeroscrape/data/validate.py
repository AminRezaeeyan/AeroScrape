import logging

import pandas as pd
import pandera.pandas as pa
from pandera.errors import SchemaError

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
