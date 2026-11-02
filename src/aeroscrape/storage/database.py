import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any

import jdatetime
import pandas as pd
import psycopg2

from aeroscrape.config import get_db_settings

logger = logging.getLogger(__name__)

PERSIAN_WEEKDAYS = ["شنبه", "یکشنبه", "دو شنبه", "سه شنبه", "چهارشنبه", "پنجشنبه", "جمعه"]

# Weekday names appear with and without spaces or zero-width non-joiners ("دوشنبه", "دو شنبه",
# "سه‌شنبه") and with Arabic or Persian forms of ی and ک, so compare them in a canonical form.
_PERSIAN_CANONICAL = str.maketrans({"ي": "ی", "ك": "ک", "\u200c": None, " ": None})


def _canonical_persian(text: str) -> str:
    return text.translate(_PERSIAN_CANONICAL)


_WEEKDAY_INDEX = {_canonical_persian(name): i for i, name in enumerate(PERSIAN_WEEKDAYS)}


# Columns that are NOT NULL in both flight tables; "place" is origin for arrivals, destination for departures
REQUIRED_FLIGHT_FIELDS = ("airline", "flight_number", "place", "status")

_UPSERT_TEMPLATE = """
    INSERT INTO {table} (
        scheduled_time, airline, flight_number, {place},
        status, counter, actual_time, register,
        aircraft, airport_id, is_international
    ) VALUES (
        %(scheduled_time)s, %(airline)s, %(flight_number)s, %(place)s,
        %(status)s, %(counter)s, %(actual_time)s, %(register)s,
        %(aircraft)s, %(airport_id)s, %(is_international)s
    )
    ON CONFLICT (flight_number, scheduled_time, airport_id)
    DO UPDATE SET
        status = EXCLUDED.status,
        counter = EXCLUDED.counter,
        actual_time = EXCLUDED.actual_time,
        register = EXCLUDED.register,
        aircraft = EXCLUDED.aircraft,
        is_international = EXCLUDED.is_international,
        {place} = EXCLUDED.{place},
        airline = EXCLUDED.airline
"""
UPSERT_SQL = {
    "arrivals": _UPSERT_TEMPLATE.format(table="arrivals", place="origin"),
    "departures": _UPSERT_TEMPLATE.format(table="departures", place="destination"),
}


def _clean(value: Any) -> str | None:
    """Strip a cell value, mapping empty strings and NaN to None."""
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return None
    text = str(value).strip()
    return text or None


@dataclass
class InsertResult:
    total: int = 0
    inserted: int = 0
    failed: int = 0
    skipped: Counter[str] = field(default_factory=Counter)

    def summary(self) -> str:
        skipped = ", ".join(f"{reason}: {n}" for reason, n in self.skipped.most_common()) or "none"
        return f"{self.inserted}/{self.total} upserted, {self.failed} rejected by the database, skipped ({skipped})"


class Database:
    def __init__(self):
        db_settings = get_db_settings()

        self.conn_params = {
            "dbname": db_settings["DB_NAME"],
            "user": db_settings["DB_USER"],
            "password": db_settings["DB_PASSWORD"],
            "host": db_settings["DB_HOST"],
            "port": db_settings["DB_PORT"],
        }

    def get_connection(self):
        return psycopg2.connect(**self.conn_params)

    def test_connection(self) -> bool:
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
                    return True
        except Exception as e:
            logger.error(f"Database connection failed: {str(e)}")
            return False

    def _parse_time_with_weekday(
        self, time_str_input: str, base_date: datetime | None = None
    ) -> tuple[datetime | None, str | None]:
        """
        Convert time string with Persian weekday or full Persian date to datetime.
        Returns (datetime_obj, source_flag) where source_flag indicates how date was determined.
        source_flag can be: 'full_date_parsed', 'weekday_parsed', 'time_only_fallback', or None.
        """
        if not time_str_input or not isinstance(time_str_input, str):
            logging.debug(f"_parse_time_with_weekday: Invalid time_str_input: '{time_str_input}'.")
            return None, None

        original_time_str = time_str_input.strip()

        persian_date_match = re.search(r"(\d{4}-\d{1,2}-\d{1,2})", original_time_str)
        time_match_general = re.search(r"(\d{1,2}:\d{1,2})", original_time_str)

        if persian_date_match:
            persian_date_str = persian_date_match.group(1)
            time_str_for_full_date = time_match_general.group(1) if time_match_general else None

            if time_str_for_full_date and time_str_for_full_date in persian_date_str:
                time_str_for_full_date = None

            if time_str_for_full_date:  # Only proceed if both date and a plausible separate time are found
                try:
                    year, month, day = map(int, persian_date_str.split("-"))
                    if 1300 <= year <= 1500:
                        gregorian_date = jdatetime.date(year, month, day).togregorian()
                    elif 1900 <= year <= 2100:
                        # Already Gregorian, e.g. CSV files written by export_csv
                        gregorian_date = date(year, month, day)
                    else:
                        raise ValueError("Year is neither a plausible Persian nor Gregorian year.")

                    clean_time_str = time_str_for_full_date.split("/")[0]
                    if clean_time_str.count(":") > 1:
                        clean_time_str = ":".join(clean_time_str.split(":")[:2])

                    time_obj = datetime.strptime(clean_time_str, "%H:%M").time()
                    dt_result = datetime.combine(gregorian_date, time_obj)
                    logging.debug(f"Parsed as 'full_date_parsed': '{original_time_str}' -> {dt_result}")
                    return dt_result, "full_date_parsed"
                except Exception as e:
                    logging.debug(
                        f"Failed to parse as 'full_date_parsed' (regex) from '{original_time_str}': {str(e)}. Trying other methods."
                    )
            else:
                logging.debug(
                    f"Found Persian date '{persian_date_str}' but no clear time in '{original_time_str}'. Trying weekday parsing."
                )

        parsed_weekday_from_str = None
        persian_weekday_index_from_input = None
        remainder_str_after_weekday = original_time_str

        leading_word = re.match(r"\s*(\D+?)\s*(?=\d)", original_time_str)
        if leading_word:
            persian_weekday_index_from_input = _WEEKDAY_INDEX.get(_canonical_persian(leading_word.group(1)))
        if leading_word and persian_weekday_index_from_input is not None:
            parsed_weekday_from_str = PERSIAN_WEEKDAYS[persian_weekday_index_from_input]
            remainder_str_after_weekday = original_time_str[leading_word.end() :]

        if parsed_weekday_from_str and persian_weekday_index_from_input is not None:
            time_match_after_weekday = re.search(r"(\d{1,2}:\d{1,2})", remainder_str_after_weekday)
            if time_match_after_weekday:
                hh_mm_str = time_match_after_weekday.group(1)
                try:
                    time_obj = datetime.strptime(hh_mm_str, "%H:%M").time()
                    reference_gregorian_date = base_date.date() if base_date else datetime.now().date()

                    gregorian_weekday_of_ref_date = reference_gregorian_date.weekday()
                    persian_weekday_index_of_ref_date = (gregorian_weekday_of_ref_date + 2) % 7
                    raw_days_difference = (persian_weekday_index_from_input - persian_weekday_index_of_ref_date + 7) % 7

                    target_days_offset = 0
                    if raw_days_difference == 0:
                        target_days_offset = 0
                    elif raw_days_difference == 1:
                        target_days_offset = 1
                    elif raw_days_difference == 2:
                        target_days_offset = 2
                    elif raw_days_difference == 6:
                        target_days_offset = -1
                    elif raw_days_difference == 5:
                        target_days_offset = -2
                    else:
                        logging.warning(
                            f"Invalid weekday diff for '{original_time_str}'. Weekday: '{parsed_weekday_from_str}'."
                        )
                        return None, None

                    target_gregorian_date = reference_gregorian_date + timedelta(days=target_days_offset)
                    dt_result = datetime.combine(target_gregorian_date, time_obj)
                    logging.debug(f"Parsed as 'weekday_parsed': '{original_time_str}' -> {dt_result}")
                    return dt_result, "weekday_parsed"
                except ValueError:
                    logging.warning(
                        f"Extracted weekday '{parsed_weekday_from_str}' but failed to parse time '{hh_mm_str}' from '{original_time_str}'."
                    )
                except Exception as e:
                    logging.error(
                        f"Error processing weekday '{parsed_weekday_from_str}' and time '{hh_mm_str}' from '{original_time_str}': {str(e)}"
                    )
                    return None, None
            else:
                logging.debug(
                    f"Found weekday '{parsed_weekday_from_str}' but no HH:MM in remainder '{remainder_str_after_weekday}'."
                )

        logging.debug(f"No full date or leading weekday in '{original_time_str}'. Falling back to _parse_time.")
        return self._parse_time(original_time_str, base_date)

    def _parse_time(self, time_str: str, base_date: datetime | None = None) -> tuple[datetime | None, str | None]:
        """Convert time string (extracting HH:MM with regex) to datetime, using base_date for the date part."""
        if not time_str or not isinstance(time_str, str):
            return None, None

        time_match = re.search(r"(\d{1,2}:\d{1,2})", time_str)

        if not time_match:
            logging.warning(f"_parse_time: No HH:MM pattern in '{time_str}'.")
            return None, None

        time_str_component = time_match.group(1)
        time_str_component = time_str_component.split("/")[0]
        if time_str_component.count(":") > 1:
            time_str_component = ":".join(time_str_component.split(":")[:2])

        try:
            time_obj = datetime.strptime(time_str_component, "%H:%M").time()
            reference_date_part = base_date.date() if base_date else datetime.now().date()
            dt_result = datetime.combine(reference_date_part, time_obj)
            logging.debug(
                f"Parsed as 'time_only_fallback': '{time_str}' (extracted HH:MM '{time_str_component}') with base_date {reference_date_part.strftime('%Y-%m-%d')} -> {dt_result}"
            )
            return dt_result, "time_only_fallback"
        except ValueError as e:
            logging.warning(
                f"_parse_time: Failed to parse extracted time '{time_str_component}' from '{time_str}': {str(e)}"
            )
            return None, None

    def _get_or_create_airport(self, cur, airport_name: str) -> int:
        """Return the airport's ID, creating it inside the caller's transaction if needed."""
        cur.execute("SELECT id FROM airports WHERE name = %s", (airport_name,))
        result = cur.fetchone()
        if result:
            return result[0]
        cur.execute("INSERT INTO airports (name) VALUES (%s) RETURNING id", (airport_name,))
        return cur.fetchone()[0]

    def _prepare_flight(self, row: pd.Series, base_date: datetime | None) -> tuple[str, dict[str, Any]] | str:
        """Turn a scraped or imported row into (table, values), or return why it must be skipped."""
        scheduled_time_str = _clean(row.get("scheduled_time"))
        actual_time_str = _clean(row.get("actual_time"))

        scheduled_time, _ = (
            self._parse_time_with_weekday(scheduled_time_str, base_date) if scheduled_time_str else (None, None)
        )
        # An HH:MM-only actual time belongs to the flight's own day, not the day it was scraped
        actual_time, actual_time_date_source = (
            self._parse_time_with_weekday(actual_time_str, scheduled_time or base_date)
            if actual_time_str
            else (None, None)
        )

        # ...unless it is far earlier than the schedule, which means it rolled past midnight
        if (
            scheduled_time
            and actual_time
            and actual_time_date_source == "time_only_fallback"
            and actual_time < scheduled_time - timedelta(hours=3)
        ):
            actual_time += timedelta(days=1)

        flight_type = str(row.get("flight_type", "")).lower()
        is_arrival = "arrival" in flight_type
        values = {
            "scheduled_time": scheduled_time,
            "airline": _clean(row.get("airline")),
            "flight_number": _clean(row.get("flight_number")),
            "place": _clean(row.get("origin" if is_arrival else "destination")),
            "status": _clean(row.get("status")),
            "counter": _clean(row.get("counter")),
            "actual_time": actual_time,
            "register": _clean(row.get("register")),
            "aircraft": _clean(row.get("aircraft")),
            "airport": _clean(row.get("airport")) or "Unknown Airport",
            "is_international": "international" in flight_type,
        }

        if scheduled_time is None:
            return "unparseable scheduled_time" if scheduled_time_str else "missing scheduled_time"
        for name in REQUIRED_FLIGHT_FIELDS:
            if not values[name]:
                return f"missing {name}"
        return ("arrivals" if is_arrival else "departures"), values

    def insert_flights_from_dataframe(self, df: pd.DataFrame, base_date: datetime | None = None) -> InsertResult:
        """Upsert flights from a DataFrame.

        Each row runs in its own savepoint, so a row the database rejects is rolled
        back alone instead of aborting the transaction for every row after it.
        """
        result = InsertResult(total=len(df))
        airport_ids: dict[str, int] = {}

        with self.get_connection() as conn:
            with conn.cursor() as cur:
                for _, row in df.iterrows():
                    prepared = self._prepare_flight(row, base_date)
                    if isinstance(prepared, str):
                        result.skipped[prepared] += 1
                        logger.debug("Skipping flight %s: %s", row.get("flight_number"), prepared)
                        continue
                    table, values = prepared

                    cur.execute("SAVEPOINT flight_row")
                    try:
                        airport_id = airport_ids.get(values["airport"]) or self._get_or_create_airport(
                            cur, values["airport"]
                        )
                        cur.execute(UPSERT_SQL[table], {**values, "airport_id": airport_id})
                        cur.execute("RELEASE SAVEPOINT flight_row")
                    except psycopg2.Error as db_err:
                        cur.execute("ROLLBACK TO SAVEPOINT flight_row")
                        result.failed += 1
                        logger.error("Database rejected flight %s: %s", values["flight_number"], db_err)
                        continue
                    airport_ids[values["airport"]] = airport_id
                    result.inserted += 1

        logger.info("Flight upsert finished: %s", result.summary())
        return result

    def insert_flights_from_csv(self, csv_path: str, base_date: datetime | None = None) -> InsertResult:
        """Insert flight data from a CSV file into the database."""
        try:
            df = pd.read_csv(csv_path, dtype=str)
            df = df.fillna("")

            if base_date is not None:
                gregorian_weekday_of_base = base_date.weekday()
                persian_weekday_of_base = PERSIAN_WEEKDAYS[(gregorian_weekday_of_base + 2) % 7]

                def _prepend_base_weekday_if_needed(time_val_from_csv: Any) -> str:
                    time_str = str(time_val_from_csv).strip()
                    if not time_str:
                        return ""

                    if re.search(r"\d{4}-\d{1,2}-\d{1,2}", time_str):
                        return time_str
                    if any(persian_wd in time_str for persian_wd in PERSIAN_WEEKDAYS):
                        return time_str

                    if ":" in time_str:
                        logging.debug(
                            f"Prepending '{persian_weekday_of_base}' to time '{time_str}' from CSV for scheduled_time based on base_date {base_date.strftime('%Y-%m-%d')}"
                        )
                        return f"{persian_weekday_of_base} {time_str}"

                    return time_str

                if "scheduled_time" in df.columns:
                    df["scheduled_time"] = df["scheduled_time"].apply(_prepend_base_weekday_if_needed)

            expected_cols = [
                "airport",
                "scheduled_time",
                "actual_time",
                "flight_type",
                "airline",
                "flight_number",
                "destination_or_origin",
                "status",
                "counter",
                "register",
                "aircraft",
            ]
            for col in expected_cols:
                if col not in df.columns:
                    df[col] = ""

            df["destination"] = df.apply(
                lambda r: (
                    r.get("destination_or_origin", "") if "departure" in str(r.get("flight_type", "")).lower() else ""
                ),
                axis=1,
            )
            df["origin"] = df.apply(
                lambda r: (
                    r.get("destination_or_origin", "") if "arrival" in str(r.get("flight_type", "")).lower() else ""
                ),
                axis=1,
            )

            for col in ["scheduled_time", "actual_time"]:
                if col in df.columns:
                    df[col] = df[col].replace("", None)

            return self.insert_flights_from_dataframe(df, base_date)

        except FileNotFoundError:
            logging.error(f"CSV file not found at path: {csv_path}")
            return InsertResult()
        except pd.errors.EmptyDataError:
            logging.error(f"CSV file is empty: {csv_path}")
            return InsertResult()
        except Exception as e:
            logging.error(f"Error importing CSV file '{csv_path}': {str(e)}", exc_info=True)
            return InsertResult()

    def _format_datetime_to_persian_weekday_time(self, dt_obj: datetime | None) -> str:
        """Helper function to format Gregorian datetime to 'Persian Weekday HH:MM' string."""
        if dt_obj is None:
            return ""
        try:
            j_datetime_obj = jdatetime.datetime.fromgregorian(datetime=dt_obj)
            weekday_name = PERSIAN_WEEKDAYS[j_datetime_obj.weekday()]
            time_part = j_datetime_obj.strftime("%H:%M")
            return f"{weekday_name} {time_part}"
        except Exception as e:
            logging.error(f"Error formatting datetime {dt_obj} to Persian weekday time: {e}")
            return ""

    def fetch_flights(self, start: datetime | None = None, end: datetime | None = None) -> pd.DataFrame:
        """Flights scheduled in [start, end), in the scraped CSV format the cleaner reads.

        Training reads this format from an export; the API's feature lookup reads it straight
        from here, so both go through the same cleaning and feature code.
        """
        conditions, params = [], []
        if start is not None:
            conditions.append("f.scheduled_time >= %s")
            params.append(start)
        if end is not None:
            conditions.append("f.scheduled_time < %s")
            params.append(end)
        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        query = f"""
            SELECT f.scheduled_time, f.airline, f.flight_number, f.place AS destination_or_origin,
                   f.status, f.counter, f.actual_time, f.register, f.aircraft,
                   ap.name AS airport, f.flight_type
            FROM (
                SELECT scheduled_time, airline, flight_number, origin AS place, status, counter, actual_time,
                       register, aircraft, airport_id,
                       CASE WHEN is_international THEN 'International Arrival' ELSE 'Domestic Arrival' END AS flight_type
                FROM arrivals
                UNION ALL
                SELECT scheduled_time, airline, flight_number, destination, status, counter, actual_time,
                       register, aircraft, airport_id,
                       CASE WHEN is_international THEN 'International Departure' ELSE 'Domestic Departure' END
                FROM departures
            ) f
            LEFT JOIN airports ap ON f.airport_id = ap.id
            {where}
        """
        with self.get_connection() as conn, conn.cursor() as cur:
            cur.execute(query, params)
            columns = [c.name for c in cur.description]
            df = pd.DataFrame(cur.fetchall(), columns=columns)
        for col in ("scheduled_time", "actual_time"):
            df[col] = pd.to_datetime(df[col]).dt.strftime("%Y-%m-%d %H:%M:%S")
        return df.astype(object).where(df.notna(), None)

    def export_csv(self, filepath: str) -> bool:
        """Export arrivals and departures to one CSV file with Gregorian datetimes (YYYY-MM-DD HH:MM:SS)."""
        try:
            df = self.fetch_flights()
            df.to_csv(filepath, index=False, encoding="utf-8-sig")
            logging.info(f"Successfully exported {len(df)} flight records to {filepath}")
            return True
        except psycopg2.Error as db_err:
            logging.error(f"Database error during CSV export: {db_err}", exc_info=True)
            return False
        except OSError as io_err:
            logging.error(f"File I/O error during CSV export to {filepath}: {io_err}", exc_info=True)
            return False
