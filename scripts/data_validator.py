import pandas as pd
import pandera.pandas as pa
from pandera.errors import SchemaError
import logging

logger = logging.getLogger(__name__)

# Define the schema for the raw data
raw_data_schema = pa.DataFrameSchema({
    "scheduled_time": pa.Column(pa.String, nullable=False),
    "actual_time": pa.Column(pa.String, nullable=True),
    "airline": pa.Column(pa.String, nullable=False),
    "flight_number": pa.Column(pa.String, nullable=False),
    "destination_or_origin": pa.Column(pa.String, nullable=False),
    "status": pa.Column(pa.String, nullable=False),
    "flight_type": pa.Column(pa.String, pa.Check.isin(['Domestic Departure', 'Domestic Arrival', 'International Arrival', 'International Departure'])),
}, strict=False) # strict=False means other columns can exist

def validate_raw_data(data_path):
    """
    Loads raw data and validates it against the Pandera schema.
    Returns True if valid, raises SchemaError otherwise.
    """
    logger.info(f"Validating raw data from {data_path}")
    try:
        df = pd.read_csv(data_path)
        raw_data_schema.validate(df, lazy=True) # lazy=True collects all errors
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