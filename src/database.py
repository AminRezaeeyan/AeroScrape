import psycopg2
from psycopg2.extras import DictCursor
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
import pandas as pd
from config import Config
import logging

PERSIAN_WEEKDAYS = ['شنبه', 'يكشنبه', 'دوشنبه', 'سه شنبه', 'چهارشنبه', 'پنجشنبه', 'جمعه']

class Database:
    def __init__(self):
        self.conn_params = {
            'dbname': Config.DB_NAME,
            'user': Config.DB_USER,
            'password': Config.DB_PASSWORD,
            'host': Config.DB_HOST,
            'port': Config.DB_PORT
        }

    def get_connection(self):
        return psycopg2.connect(**self.conn_params)

    def test_connection(self) -> bool:
        """Test the database connection"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
                    return True
        except Exception as e:
            logging.error(f"Database connection failed: {str(e)}")
            return False

    def _parse_time_with_weekday(self, time_str: str, base_date: datetime = None) -> Optional[datetime]:
        """Convert time string with Persian weekday to datetime."""
        if not time_str or not isinstance(time_str, str):
            return None
            
        parts = time_str.split()
        if not parts:
            return None

        weekday = parts[0]
        if weekday in PERSIAN_WEEKDAYS:
            time_str = parts[-1]
            time_str = time_str.split('/')[0]
            
            if time_str.count(':') > 1:
                time_str = ':'.join(time_str.split(':')[:2])
            
            try:
                time_part = datetime.strptime(time_str, '%H:%M').time()
                
                # Use base_date if provided, otherwise use today
                reference_date = base_date.date() if base_date else datetime.now().date()
                
                # Get reference date's Persian weekday index
                ref_weekday_index = reference_date.weekday()
                ref_persian_index = (ref_weekday_index + 1) % 7
                input_weekday_index = PERSIAN_WEEKDAYS.index(weekday)
                
                # Calculate days to add
                days_to_add = (input_weekday_index - ref_persian_index) % 7
                
                return datetime.combine(reference_date + timedelta(days=days_to_add), time_part)
            except Exception as e:
                logging.warning(f"Failed to parse time '{time_str}': {str(e)}")
                return None
        else:
            # If no weekday is specified, assume it's the reference date
            return self._parse_time(time_str, base_date)

    def _parse_time(self, time_str: str, base_date: datetime = None) -> Optional[datetime]:
        """Convert time string to datetime."""
        if not time_str or not isinstance(time_str, str):
            return None
            
        time_parts = time_str.split()
        if not time_parts:
            return None
            
        time_str = time_parts[-1]
        
        time_str = time_str.split('/')[0]
        
        if time_str.count(':') > 1:
            time_str = ':'.join(time_str.split(':')[:2])
            
        try:
            time_part = datetime.strptime(time_str, '%H:%M').time()
            reference_date = base_date.date() if base_date else datetime.now().date()
            return datetime.combine(reference_date, time_part)
        except Exception as e:
            logging.warning(f"Failed to parse time '{time_str}': {str(e)}")
            return None

    def _get_or_create_airport(self, airport_name: str) -> int:
        """Get or create airport and return its ID"""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                # First try to get the existing airport
                cur.execute("""
                    SELECT id FROM airports WHERE name = %s
                """, (airport_name,))
                result = cur.fetchone()
                
                if result:
                    return result[0]
                
                # If not found, insert new airport
                cur.execute("""
                    INSERT INTO airports (name)
                    VALUES (%s)
                    RETURNING id
                """, (airport_name,))
                return cur.fetchone()[0]

    def insert_flights_from_dataframe(self, df: pd.DataFrame, base_date: datetime = None) -> int:
        """Insert flight data from a pandas DataFrame into the database"""
        inserted_count = 0
        
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                for _, row in df.iterrows():
                    try:
                        airport_id = self._get_or_create_airport(row['airport'])
                        scheduled_time = self._parse_time_with_weekday(row['scheduled_time'], base_date)
                        actual_time = self._parse_time_with_weekday(row['actual_time'], base_date) if pd.notna(row['actual_time']) else None
                        
                        # Determine if it's an arrival or departure
                        is_arrival = 'arrival' in row['flight_type'].lower()
                        
                        if is_arrival:
                            # Insert into arrivals table
                            cur.execute("""
                                INSERT INTO arrivals (
                                    scheduled_time, airline, flight_number, origin,
                                    status, counter, actual_time, register,
                                    aircraft, airport_id
                                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                                ON CONFLICT (flight_number, scheduled_time, airport_id)
                                DO UPDATE SET
                                    status = EXCLUDED.status,
                                    counter = EXCLUDED.counter,
                                    actual_time = EXCLUDED.actual_time,
                                    register = EXCLUDED.register,
                                    aircraft = EXCLUDED.aircraft
                            """, (
                                scheduled_time, row['airline'], row['flight_number'],
                                row['origin'], row['status'], row['counter'],
                                actual_time, row['register'], row['aircraft'],
                                airport_id
                            ))
                        else:
                            # Insert into departures table
                            cur.execute("""
                                INSERT INTO departures (
                                    scheduled_time, airline, flight_number, destination,
                                    status, counter, actual_time, register,
                                    aircraft, airport_id
                                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                                ON CONFLICT (flight_number, scheduled_time, airport_id)
                                DO UPDATE SET
                                    status = EXCLUDED.status,
                                    counter = EXCLUDED.counter,
                                    actual_time = EXCLUDED.actual_time,
                                    register = EXCLUDED.register,
                                    aircraft = EXCLUDED.aircraft
                            """, (
                                scheduled_time, row['airline'], row['flight_number'],
                                row['destination'], row['status'], row['counter'],
                                actual_time, row['register'], row['aircraft'],
                                airport_id
                            ))
                        
                        inserted_count += 1
                        
                    except Exception as e:
                        logging.error(f"Error inserting flight {row.get('flight_number', 'UNKNOWN')}: {str(e)}")
                        continue
                        
        return inserted_count

    def insert_flights_from_csv(self, csv_path: str, base_date: datetime = None) -> int:
        """Insert flight data from a CSV file into the database.
        
        Args:
            csv_path: Path to the CSV file
            base_date: Base date for the flights (defaults to today)
        
        The CSV file should have the following columns:
        - scheduled_time: Time in HH:MM format or Persian weekday + time (e.g., "شنبه 14:30")
        - airline: Airline name
        - flight_number: Flight number
        - destination_or_origin: Origin or destination airport
        - status: Flight status
        - counter: Counter number
        - actual_time: Actual time in HH:MM format or Persian weekday + time (optional)
        - register: Aircraft registration
        - aircraft: Aircraft type
        - airport: Airport name
        - flight_type: One of ["Domestic Arrival", "Domestic Departure", "International Arrival", "International Departure"]
        """
        try:
            df = pd.read_csv(csv_path)
            
            if base_date is not None:
                base_weekday = PERSIAN_WEEKDAYS[(base_date.weekday() + 1) % 7]
                df['scheduled_time'] = df['scheduled_time'].apply(
                    lambda x: f"{base_weekday} {x}" if ':' in x else x
                )
                if 'actual_time' in df.columns:
                    df['actual_time'] = df['actual_time'].apply(
                        lambda x: f"{base_weekday} {x}" if pd.notna(x) and ':' in x else x
                    )
            
            # Split destination_or_origin into destination and origin based on flight_type
            df['destination'] = df.apply(
                lambda row: row['destination_or_origin'] if 'departure' in row['flight_type'].lower() else None,
                axis=1
            )
            df['origin'] = df.apply(
                lambda row: row['destination_or_origin'] if 'arrival' in row['flight_type'].lower() else None,
                axis=1
            )
            
            return self.insert_flights_from_dataframe(df, base_date)
            
        except Exception as e:
            logging.error(f"Error importing CSV file: {str(e)}")
            return 0