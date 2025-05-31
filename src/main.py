from scraper import get_all_flights_data
from database import Database
import logging
from datetime import datetime
import os

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
    try:
        # Initialize database connection
        db = Database()
        if not db.test_connection():
            logging.error("Failed to connect to database")
            return

        logging.info("Starting flight data scraping...")
        
        # Get flight data from scraper
        df = get_all_flights_data()
        if df.empty:
            logging.error("No flight data was scraped")
            return

        logging.info(f"Successfully scraped {len(df)} flights")
        
        # Insert flights into database
        inserted_count = db.insert_flights_from_dataframe(df)
        logging.info(f"Successfully inserted/updated {inserted_count} flights in the database")

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()
