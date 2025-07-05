import os
import sys

current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_script_dir)

# Add the project root to sys.path so modules like 'utils' and 'src' are discoverable
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    
from scraper import get_all_flights_data
from database import Database
from utils.config import get_app_config
import logging
import os
import argparse
from datetime import datetime

# Configure logging
os.makedirs('logs', exist_ok=True)
log_filename = os.path.join('logs', f'flight_ops_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def run_data_operations():
    try:
        app_config = get_app_config()
        logger.info("Application configuration successfully accessed.")
    except Exception as e:
        logger.critical(f"Fatal error during application startup due to configuration issues: {e}")
        return # Cannot proceed

    parser = argparse.ArgumentParser(description='Flight data scraper, importer, and exporter.')
    parser.add_argument('--date', type=str, help='Base date for flights (YYYY-MM-DD format) for CSV import or scraping. Defaults to current date if not provided.')

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--csv', type=str, help='Path to CSV file for import. Uses --date or current date if times in CSV are relative.')
    group.add_argument('--export-csv-path', type=str, help='Path to export the database content to a CSV file.')

    args = parser.parse_args()

    try:
        db = Database()
        if not db.test_connection():
            logger.error("Failed to connect to the database. Please check connection parameters and DB status.")
            return

        base_date = None
        if args.date:
            try:
                base_date = datetime.strptime(args.date, '%Y-%m-%d')
                logger.info(f"Using provided base date: {base_date.strftime('%Y-%m-%d')}")
            except ValueError:
                logger.error("Invalid date format for --date. Please use YYYY-MM-DD. Aborting.")
                return
        else:
            base_date = datetime.now()
            logger.info(f"No --date provided. Using current date as base date: {base_date.strftime('%Y-%m-%d')}")

        if args.export_csv_path:
            logger.info(f"Attempting to export database content to CSV: {args.export_csv_path}")
            export_dir = os.path.dirname(args.export_csv_path)
            if export_dir and not os.path.exists(export_dir):
                os.makedirs(export_dir)
            success = db.export_csv(args.export_csv_path)
            if success:
                logger.info(f"Database content successfully exported to {args.export_csv_path}")
            else:
                logger.error(f"Failed to export database content to {args.export_csv_path}")

        elif args.csv:
            logger.info(f"Attempting to import flights from CSV: {args.csv}")
            if not os.path.exists(args.csv):
                logger.error(f"CSV file not found at path: {args.csv}. Aborting import.")
                return
            inserted_count = db.insert_flights_from_csv(args.csv, base_date=base_date)
            logger.info(f"Successfully inserted/updated {inserted_count} flights from CSV file: {args.csv}")

        else: # Default behavior: scrape
            logger.info("Starting flight data scraping...")
            df = get_all_flights_data()
            if df is None or df.empty:
                logger.warning("No flight data was scraped. Nothing to import.")
                return
            logger.info(f"Successfully scraped {len(df)} flights.")
            inserted_count = db.insert_flights_from_dataframe(df, base_date=base_date)
            logger.info(f"Successfully inserted/updated {inserted_count} flights from scraping into the database.")

    except Exception as e:
        logger.error(f"An unexpected error occurred in CLI execution: {str(e)}", exc_info=True)

if __name__ == "__main__":
    run_data_operations()