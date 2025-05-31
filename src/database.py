import psycopg2
from psycopg2.extras import DictCursor
from datetime import datetime
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

    def _parse_time(self, time_str: str) -> Optional[datetime]:
        """Convert time string to datetime today."""
        if not time_str or not isinstance(time_str, str):
            return None
            
        # Extract just the time part (HH:MM)
        time_parts = time_str.split()
        if not time_parts:
            return None
            
        time_str = time_parts[-1]
        
        # Remove any suffixes after the time (e.g., /1)
        time_str = time_str.split('/')[0]
        
        # Remove seconds if present
        if time_str.count(':') > 1:
            time_str = ':'.join(time_str.split(':')[:2])
            
        try:
            time_part = datetime.strptime(time_str, '%H:%M').time()
            return datetime.combine(datetime.now().date(), time_part)
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

    def insert_flights_from_dataframe(self, df: pd.DataFrame) -> int:
        """Insert flight data from a pandas DataFrame into the database"""
        inserted_count = 0
        
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                for _, row in df.iterrows():
                    try:
                        # Get airport ID
                        airport_id = self._get_or_create_airport(row['airport'])
                        
                        # Parse times
                        scheduled_time = self._parse_time(row['scheduled_time'])
                        actual_time = self._parse_time(row['actual_time']) if pd.notna(row['actual_time']) else None
                        
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
