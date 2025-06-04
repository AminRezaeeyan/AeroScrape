import psycopg2
from psycopg2.extras import DictCursor
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any, Tuple
import pandas as pd
from config import Config
import logging
import jdatetime
import csv
import re

# Define PERSIAN_WEEKDAYS at the module level
PERSIAN_WEEKDAYS = ['شنبه', 'یکشنبه', 'دو شنبه', 'سه شنبه', 'چهارشنبه', 'پنجشنبه', 'جمعه']

class Database:
    def __init__(self):
        # Initialize database connection parameters from Config
        self.conn_params = {
            'dbname': Config.DB_NAME,
            'user': Config.DB_USER,
            'password': Config.DB_PASSWORD,
            'host': Config.DB_HOST,
            'port': Config.DB_PORT
        }

    def get_connection(self):
        # Establish and return a new database connection
        return psycopg2.connect(**self.conn_params)

    def test_connection(self) -> bool:
        """Test the database connection"""
        try:
            # Try to connect and execute a simple query
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
                    return True
        except Exception as e:
            logging.error(f"Database connection failed: {str(e)}")
            return False

    def _parse_time_with_weekday(self, time_str_input: str, base_date: Optional[datetime] = None) -> Tuple[Optional[datetime], Optional[str]]:
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


            if time_str_for_full_date: # Only proceed if both date and a plausible separate time are found
                try:
                    year, month, day = map(int, persian_date_str.split('-'))
                    if not (1300 <= year <= 1500): 
                        raise ValueError("Persian year out of plausible range.")
                    persian_date_obj = jdatetime.date(year, month, day)
                    gregorian_date = persian_date_obj.togregorian()
                    
                    clean_time_str = time_str_for_full_date.split('/')[0] 
                    if clean_time_str.count(':') > 1: 
                        clean_time_str = ':'.join(clean_time_str.split(':')[:2])
                    
                    time_obj = datetime.strptime(clean_time_str, "%H:%M").time()
                    dt_result = datetime.combine(gregorian_date, time_obj)
                    logging.debug(f"Parsed as 'full_date_parsed': '{original_time_str}' -> {dt_result}")
                    return dt_result, 'full_date_parsed'
                except Exception as e:
                    logging.debug(f"Failed to parse as 'full_date_parsed' (regex) from '{original_time_str}': {str(e)}. Trying other methods.")
            else:
                logging.debug(f"Found Persian date '{persian_date_str}' but no clear time in '{original_time_str}'. Trying weekday parsing.")


        sorted_persian_weekdays = sorted(PERSIAN_WEEKDAYS, key=len, reverse=True)
        parsed_weekday_from_str = None
        remainder_str_after_weekday = original_time_str

        for wd in sorted_persian_weekdays:
            if original_time_str.startswith(wd):
                parsed_weekday_from_str = wd
                remainder_str_after_weekday = original_time_str[len(wd):].strip()
                break
        
        if parsed_weekday_from_str:
            time_match_after_weekday = re.search(r"(\d{1,2}:\d{1,2})", remainder_str_after_weekday)
            if time_match_after_weekday:
                hh_mm_str = time_match_after_weekday.group(1)
                try:
                    time_obj = datetime.strptime(hh_mm_str, '%H:%M').time()
                    reference_gregorian_date = base_date.date() if base_date else datetime.now().date()
                    
                    gregorian_weekday_of_ref_date = reference_gregorian_date.weekday() 
                    persian_weekday_index_of_ref_date = (gregorian_weekday_of_ref_date + 2) % 7 
                    persian_weekday_index_from_input = PERSIAN_WEEKDAYS.index(parsed_weekday_from_str)
                    
                    raw_days_difference = (persian_weekday_index_from_input - persian_weekday_index_of_ref_date + 7) % 7 
                    
                    target_days_offset = 0
                    if raw_days_difference == 0: target_days_offset = 0
                    elif raw_days_difference == 1: target_days_offset = 1
                    elif raw_days_difference == 2: target_days_offset = 2
                    elif raw_days_difference == 6: target_days_offset = -1
                    elif raw_days_difference == 5: target_days_offset = -2
                    else:
                        logging.warning(f"Invalid weekday diff for '{original_time_str}'. Weekday: '{parsed_weekday_from_str}'.")
                        return None, None
                        
                    target_gregorian_date = reference_gregorian_date + timedelta(days=target_days_offset)
                    dt_result = datetime.combine(target_gregorian_date, time_obj)
                    logging.debug(f"Parsed as 'weekday_parsed': '{original_time_str}' -> {dt_result}")
                    return dt_result, 'weekday_parsed'
                except ValueError: 
                    logging.warning(f"Extracted weekday '{parsed_weekday_from_str}' but failed to parse time '{hh_mm_str}' from '{original_time_str}'.")
                except Exception as e:
                    logging.error(f"Error processing weekday '{parsed_weekday_from_str}' and time '{hh_mm_str}' from '{original_time_str}': {str(e)}")
                    return None, None
            else:
                logging.debug(f"Found weekday '{parsed_weekday_from_str}' but no HH:MM in remainder '{remainder_str_after_weekday}'.")

        logging.debug(f"No full date or leading weekday in '{original_time_str}'. Falling back to _parse_time.")
        return self._parse_time(original_time_str, base_date)


    def _parse_time(self, time_str: str, base_date: Optional[datetime] = None) -> Tuple[Optional[datetime], Optional[str]]:
        """Convert time string (extracting HH:MM with regex) to datetime, using base_date for the date part."""
        if not time_str or not isinstance(time_str, str):
            return None, None
            
        time_match = re.search(r"(\d{1,2}:\d{1,2})", time_str)
        
        if not time_match:
            logging.warning(f"_parse_time: No HH:MM pattern in '{time_str}'.")
            return None, None

        time_str_component = time_match.group(1)
        time_str_component = time_str_component.split('/')[0]
        if time_str_component.count(':') > 1: 
            time_str_component = ':'.join(time_str_component.split(':')[:2])
            
        try:
            time_obj = datetime.strptime(time_str_component, '%H:%M').time()
            reference_date_part = base_date.date() if base_date else datetime.now().date()
            dt_result = datetime.combine(reference_date_part, time_obj)
            logging.debug(f"Parsed as 'time_only_fallback': '{time_str}' (extracted HH:MM '{time_str_component}') with base_date {reference_date_part.strftime('%Y-%m-%d')} -> {dt_result}")
            return dt_result, 'time_only_fallback'
        except ValueError as e:
            logging.warning(f"_parse_time: Failed to parse extracted time '{time_str_component}' from '{time_str}': {str(e)}")
            return None, None

    def _get_or_create_airport(self, airport_name: str) -> int:
        """Get or create airport and return its ID"""
        if pd.isna(airport_name) or not airport_name: 
            airport_name = "Unknown" 
        elif not isinstance(airport_name, str):
            airport_name = str(airport_name)
        airport_name = airport_name.strip()

        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT id FROM airports WHERE name = %s", (airport_name,))
                result = cur.fetchone()
                if result:
                    return result[0]
                
                cur.execute("INSERT INTO airports (name) VALUES (%s) RETURNING id", (airport_name,))
                airport_id = cur.fetchone()[0]
                return airport_id

    def insert_flights_from_dataframe(self, df: pd.DataFrame, base_date: Optional[datetime] = None) -> int:
        """Insert flight data from a pandas DataFrame into the database"""
        inserted_count = 0
        
        with self.get_connection() as conn: 
            with conn.cursor() as cur:
                for _, row in df.iterrows():
                    try:
                        airport_val = row.get('airport')
                        airport_name_to_use = str(airport_val).strip() if pd.notna(airport_val) and str(airport_val).strip() else "Unknown Airport"
                        airport_id = self._get_or_create_airport(airport_name_to_use)
                        
                        scheduled_time_str = str(row.get('scheduled_time')).strip() if pd.notna(row.get('scheduled_time')) and str(row.get('scheduled_time')).strip() else None
                        actual_time_str = str(row.get('actual_time')).strip() if pd.notna(row.get('actual_time')) and str(row.get('actual_time')).strip() else None

                        scheduled_time, _ = self._parse_time_with_weekday(scheduled_time_str, base_date) if scheduled_time_str else (None, None)
                        actual_time, actual_time_date_source = self._parse_time_with_weekday(actual_time_str, base_date) if actual_time_str else (None, None)
                        
                        # --- Heuristic for actual_time adjustment if its date was from base_date fallback ---
                        if scheduled_time and actual_time and actual_time_date_source == 'time_only_fallback':
                            # If actual_time's date was derived from base_date (because actual_time_str was HH:MM only)
                            # and it's significantly earlier than scheduled_time, assume it's for the next day.
                            if actual_time < (scheduled_time - timedelta(hours=3)): # Threshold can be adjusted
                                adjusted_actual_time = actual_time + timedelta(days=1)
                                logging.info(
                                    f"Adjusted actual_time for flight {row.get('flight_number', 'N/A')} (heuristic: time_only_fallback, next day): "
                                    f"Original actual_str='{actual_time_str}', "
                                    f"Scheduled='{scheduled_time.strftime('%Y-%m-%d %H:%M')}', "
                                    f"Initial actual_parse='{actual_time.strftime('%Y-%m-%d %H:%M')}', "
                                    f"Final actual='{adjusted_actual_time.strftime('%Y-%m-%d %H:%M')}'"
                                )
                                actual_time = adjusted_actual_time
                        # --- End of actual_time adjustment heuristic ---
                        
                        if not scheduled_time: 
                            logging.warning(f"Could not parse scheduled_time '{row.get('scheduled_time')}' for flight {row.get('flight_number', 'N/A')}. Row will be inserted with NULL scheduled_time if DB allows.")

                        flight_type_val = str(row.get('flight_type', '')).lower()
                        is_arrival = 'arrival' in flight_type_val
                        is_international = 'international' in flight_type_val
                        
                        airline = str(row.get('airline')).strip() if pd.notna(row.get('airline')) and str(row.get('airline')).strip() else None
                        flight_number = str(row.get('flight_number')).strip() if pd.notna(row.get('flight_number')) and str(row.get('flight_number')).strip() else None
                        origin_val = str(row.get('origin')).strip() if pd.notna(row.get('origin')) and str(row.get('origin')).strip() else None
                        destination_val = str(row.get('destination')).strip() if pd.notna(row.get('destination')) and str(row.get('destination')).strip() else None
                        status = str(row.get('status')).strip() if pd.notna(row.get('status')) and str(row.get('status')).strip() else None
                        counter = str(row.get('counter')).strip() if pd.notna(row.get('counter')) and str(row.get('counter')).strip() else None
                        register = str(row.get('register')).strip() if pd.notna(row.get('register')) and str(row.get('register')).strip() else None
                        aircraft = str(row.get('aircraft')).strip() if pd.notna(row.get('aircraft')) and str(row.get('aircraft')).strip() else None

                        if not flight_number or not airport_id : 
                             logging.error(f"Skipping flight due to missing key components: flight_number='{flight_number}', airport_id='{airport_id}' for row: {row.to_dict()}. Scheduled time was '{scheduled_time_str}' parsed to '{scheduled_time}'.")
                             continue

                        if is_arrival:
                            cur.execute("""
                                INSERT INTO arrivals (
                                    scheduled_time, airline, flight_number, origin,
                                    status, counter, actual_time, register,
                                    aircraft, airport_id, is_international
                                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                                ON CONFLICT (flight_number, scheduled_time, airport_id)
                                DO UPDATE SET
                                    status = EXCLUDED.status,
                                    counter = EXCLUDED.counter,
                                    actual_time = EXCLUDED.actual_time,
                                    register = EXCLUDED.register,
                                    aircraft = EXCLUDED.aircraft,
                                    is_international = EXCLUDED.is_international,
                                    origin = EXCLUDED.origin,
                                    airline = EXCLUDED.airline
                            """, (
                                scheduled_time, airline, flight_number,
                                origin_val, status, counter,
                                actual_time, register, aircraft, 
                                airport_id, is_international
                            ))
                        else: 
                            cur.execute("""
                                INSERT INTO departures (
                                    scheduled_time, airline, flight_number, destination,
                                    status, counter, actual_time, register,
                                    aircraft, airport_id, is_international
                                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                                ON CONFLICT (flight_number, scheduled_time, airport_id)
                                DO UPDATE SET
                                    status = EXCLUDED.status,
                                    counter = EXCLUDED.counter,
                                    actual_time = EXCLUDED.actual_time,
                                    register = EXCLUDED.register,
                                    aircraft = EXCLUDED.aircraft,
                                    is_international = EXCLUDED.is_international,
                                    destination = EXCLUDED.destination,
                                    airline = EXCLUDED.airline
                            """, (
                                scheduled_time, airline, flight_number,
                                destination_val, status, counter,
                                actual_time, register, aircraft, 
                                airport_id, is_international
                            ))
                        inserted_count += 1
                    except psycopg2.Error as db_err:
                         logging.error(f"Database error inserting flight {row.get('flight_number', 'UNKNOWN')}: {db_err}. SQL was: {cur.query.decode() if cur.query else 'N/A'}")
                         raise 
                    except Exception as e:
                        logging.error(f"General error inserting flight {row.get('flight_number', 'UNKNOWN')} (Row data: {row.to_dict()}): {str(e)}", exc_info=True)
                        continue 
        return inserted_count

    def insert_flights_from_csv(self, csv_path: str, base_date: Optional[datetime] = None) -> int:
        """Insert flight data from a CSV file into the database."""
        try:
            df = pd.read_csv(csv_path, dtype=str) 
            df = df.fillna('') 
            
            if base_date is not None:
                gregorian_weekday_of_base = base_date.weekday()
                persian_weekday_of_base = PERSIAN_WEEKDAYS[(gregorian_weekday_of_base + 2) % 7]

                def _prepend_base_weekday_if_needed(time_val_from_csv: Any) -> str:
                    time_str = str(time_val_from_csv).strip()
                    if not time_str: 
                        return '' 
                    
                    if re.search(r"\d{4}-\d{1,2}-\d{1,2}", time_str): 
                        return time_str 
                    if any(persian_wd in time_str for persian_wd in PERSIAN_WEEKDAYS):
                        return time_str

                    if ':' in time_str:
                        logging.debug(f"Prepending '{persian_weekday_of_base}' to time '{time_str}' from CSV for scheduled_time based on base_date {base_date.strftime('%Y-%m-%d')}")
                        return f"{persian_weekday_of_base} {time_str}"
                    
                    return time_str 

                if 'scheduled_time' in df.columns:
                    df['scheduled_time'] = df['scheduled_time'].apply(_prepend_base_weekday_if_needed)
            
            expected_cols = ['airport', 'scheduled_time', 'actual_time', 'flight_type', 'airline', 
                             'flight_number', 'destination_or_origin', 'status', 'counter', 'register', 'aircraft']
            for col in expected_cols:
                if col not in df.columns:
                    df[col] = '' 

            df['destination'] = df.apply(
                lambda r: r.get('destination_or_origin', '') if 'departure' in str(r.get('flight_type', '')).lower() else '',
                axis=1
            )
            df['origin'] = df.apply(
                lambda r: r.get('destination_or_origin', '') if 'arrival' in str(r.get('flight_type', '')).lower() else '',
                axis=1
            )
            
            for col in ['scheduled_time', 'actual_time']:
                if col in df.columns:
                    df[col] = df[col].replace('', None)


            return self.insert_flights_from_dataframe(df, base_date)
            
        except FileNotFoundError:
            logging.error(f"CSV file not found at path: {csv_path}")
            return 0
        except pd.errors.EmptyDataError:
            logging.error(f"CSV file is empty: {csv_path}")
            return 0
        except Exception as e:
            logging.error(f"Error importing CSV file '{csv_path}': {str(e)}", exc_info=True)
            return 0

    def _format_datetime_to_persian_weekday_time(self, dt_obj: Optional[datetime]) -> str:
        """Helper function to format Gregorian datetime to 'Persian Weekday HH:MM' string."""
        if dt_obj is None:
            return ''
        try:
            j_datetime_obj = jdatetime.datetime.fromgregorian(datetime=dt_obj)
            weekday_name = PERSIAN_WEEKDAYS[j_datetime_obj.weekday()]
            time_part = j_datetime_obj.strftime("%H:%M")
            return f"{weekday_name} {time_part}"
        except Exception as e:
            logging.error(f"Error formatting datetime {dt_obj} to Persian weekday time: {e}")
            return '' 

    def export_csv(self, filepath: str) -> bool:
        """
        Exports data from arrivals and departures tables to a single CSV file.
        Scheduled and actual times are exported as Gregorian datetime strings (YYYY-MM-DD HH:MM:SS).
        """
        all_flight_records = []
        
        csv_column_names = [
            'scheduled_time', 'airline', 'flight_number', 'destination_or_origin', 
            'status', 'counter', 'actual_time', 'register', 'aircraft', 'airport', 'flight_type'
        ]

        try:
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=DictCursor) as cur_arr:
                    cur_arr.execute("""
                        SELECT
                            a.scheduled_time, a.airline, a.flight_number, a.origin,
                            a.status, a.counter, a.actual_time, a.register,
                            a.aircraft, ap.name as airport_name, a.is_international
                        FROM arrivals a
                        LEFT JOIN airports ap ON a.airport_id = ap.id
                    """)
                    for row in cur_arr.fetchall():
                        flight_type_str = "International Arrival" if row['is_international'] else "Domestic Arrival"
                        record = {
                            'scheduled_time': row['scheduled_time'].strftime('%Y-%m-%d %H:%M:%S') if row['scheduled_time'] else '',
                            'airline': row['airline'] if row['airline'] is not None else '',
                            'flight_number': row['flight_number'] if row['flight_number'] is not None else '',
                            'destination_or_origin': row['origin'] if row['origin'] is not None else '',
                            'status': row['status'] if row['status'] is not None else '',
                            'counter': str(row['counter']) if row['counter'] is not None else '', 
                            'actual_time': row['actual_time'].strftime('%Y-%m-%d %H:%M:%S') if row['actual_time'] else '',
                            'register': row['register'] if row['register'] is not None else '',
                            'aircraft': row['aircraft'] if row['aircraft'] is not None else '',
                            'airport': row['airport_name'] if row['airport_name'] is not None else '',
                            'flight_type': flight_type_str
                        }
                        all_flight_records.append(record)
                
                with conn.cursor(cursor_factory=DictCursor) as cur_dep:
                    cur_dep.execute("""
                        SELECT
                            d.scheduled_time, d.airline, d.flight_number, d.destination,
                            d.status, d.counter, d.actual_time, d.register,
                            d.aircraft, ap.name as airport_name, d.is_international
                        FROM departures d
                        LEFT JOIN airports ap ON d.airport_id = ap.id
                    """)
                    for row in cur_dep.fetchall():
                        flight_type_str = "International Departure" if row['is_international'] else "Domestic Departure"
                        record = {
                            'scheduled_time': row['scheduled_time'].strftime('%Y-%m-%d %H:%M:%S') if row['scheduled_time'] else '',
                            'airline': row['airline'] if row['airline'] is not None else '',
                            'flight_number': row['flight_number'] if row['flight_number'] is not None else '',
                            'destination_or_origin': row['destination'] if row['destination'] is not None else '',
                            'status': row['status'] if row['status'] is not None else '',
                            'counter': str(row['counter']) if row['counter'] is not None else '', 
                            'actual_time': row['actual_time'].strftime('%Y-%m-%d %H:%M:%S') if row['actual_time'] else '',
                            'register': row['register'] if row['register'] is not None else '',
                            'aircraft': row['aircraft'] if row['aircraft'] is not None else '',
                            'airport': row['airport_name'] if row['airport_name'] is not None else '',
                            'flight_type': flight_type_str
                        }
                        all_flight_records.append(record)

            if not all_flight_records:
                logging.info("No flight data found to export.")
                df_export = pd.DataFrame(columns=csv_column_names)
            else:
                df_export = pd.DataFrame(all_flight_records, columns=csv_column_names)
            
            df_export.to_csv(filepath, index=False, encoding='utf-8-sig') 
            logging.info(f"Successfully exported {len(all_flight_records)} flight records to {filepath}")
            return True

        except psycopg2.Error as db_err:
            logging.error(f"Database error during CSV export: {db_err}", exc_info=True)
            return False
        except IOError as io_err:
            logging.error(f"File I/O error during CSV export to {filepath}: {io_err}", exc_info=True)
            return False
        except Exception as e:
            logging.error(f"An unexpected error occurred during CSV export: {e}", exc_info=True)
            return False
