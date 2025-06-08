from scraper import get_all_flights_data
from database import Database
import logging
from datetime import datetime
import os
import argparse

# Create logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)

# Configure logging
log_filename = os.path.join('logs', f'flight_ops_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)

def main():
    parser = argparse.ArgumentParser(description='Flight data scraper, importer, and exporter.')
    parser.add_argument('--date', type=str, help='Base date for flights (YYYY-MM-DD format) for CSV import or scraping. Defaults to current date if not provided.')
    
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--csv', type=str, help='Path to CSV file for import. Uses --date or current date if times in CSV are relative.')
    group.add_argument('--export-csv-path', type=str, help='Path to export the database content to a CSV file.')
    # If neither --csv nor --export_csv_path is given, it will default to scraping

    args = parser.parse_args()

    try:
        # Initialize database connection
        db = Database()
        if not db.test_connection():
            logging.error("Failed to connect to the database. Please check connection parameters and DB status.")
            return

        # Parse or set base date
        base_date = None
        if args.date:
            try:
                base_date = datetime.strptime(args.date, '%Y-%m-%d')
                logging.info(f"Using provided base date: {base_date.strftime('%Y-%m-%d')}")
            except ValueError:
                logging.error("Invalid date format for --date. Please use YYYY-MM-DD. Aborting.")
                return
        else:
            # Default to current date if --date is not provided
            base_date = datetime.now()
            logging.info(f"No --date provided. Using current date as base date: {base_date.strftime('%Y-%m-%d')}")
        
        # --- Operation Mode Logic ---

        if args.export_csv_path:
            # Mode: Export data from database to CSV
            logging.info(f"Attempting to export database content to CSV: {args.export_csv_path}")
            success = db.export_csv(args.export_csv_path)
            if success:
                logging.info(f"Database content successfully exported to {args.export_csv_path}")
            else:
                logging.error(f"Failed to export database content to {args.export_csv_path}")

        elif args.csv:
            # Mode: Import data from CSV into database
            logging.info(f"Attempting to import flights from CSV: {args.csv}")
            if not os.path.exists(args.csv):
                logging.error(f"CSV file not found at path: {args.csv}. Aborting import.")
                return
            inserted_count = db.insert_flights_from_csv(args.csv, base_date=base_date)
            logging.info(f"Successfully inserted/updated {inserted_count} flights from CSV file: {args.csv}")

        else:
            # Mode: Scrape data and import into database (default if no other mode is specified)
            logging.info("Starting flight data scraping...")
        
            df = get_all_flights_data() 
            
            if df is None or df.empty:
                logging.warning("No flight data was scraped. Nothing to import.")
                return

            logging.info(f"Successfully scraped {len(df)} flights.")
            inserted_count = db.insert_flights_from_dataframe(df, base_date=base_date)
            logging.info(f"Successfully inserted/updated {inserted_count} flights from scraping into the database.")

    except Exception as e:
        logging.error(f"An unexpected error occurred in main execution: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()
