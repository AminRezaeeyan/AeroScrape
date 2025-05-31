from scraper import get_all_flights_data
from database import Database
import logging
from datetime import datetime
import os
import argparse

# Create logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join('logs', f'scraper_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')),
        logging.StreamHandler()
    ]
)

def main():
    parser = argparse.ArgumentParser(description='Flight data scraper and importer')
    parser.add_argument('--date', type=str, help='Base date for flights (YYYY-MM-DD format)')
    parser.add_argument('--csv', type=str, help='Path to CSV file for import')
    args = parser.parse_args()

    try:
        # Initialize database connection
        db = Database()
        if not db.test_connection():
            logging.error("Failed to connect to database")
            return

        # Parse base date if provided
        base_date = None
        if args.date:
            try:
                base_date = datetime.strptime(args.date, '%Y-%m-%d')
                logging.info(f"Using base date: {base_date.strftime('%Y-%m-%d')}")
            except ValueError:
                logging.error("Invalid date format. Please use YYYY-MM-DD")
                return

        if args.csv:
            # Import from CSV
            logging.info(f"Importing flights from CSV: {args.csv}")
            inserted_count = db.insert_flights_from_csv(args.csv, base_date=base_date)
            logging.info(f"Successfully inserted/updated {inserted_count} flights from CSV")
        else:
            # Scrape and import
            logging.info("Starting flight data scraping...")
            df = get_all_flights_data()
            if df.empty:
                logging.error("No flight data was scraped")
                return

            logging.info(f"Successfully scraped {len(df)} flights")
            inserted_count = db.insert_flights_from_dataframe(df, base_date=base_date)
            logging.info(f"Successfully inserted/updated {inserted_count} flights in the database")

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()
