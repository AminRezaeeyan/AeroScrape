import argparse
import logging
import os
import sys
from datetime import datetime

from aeroscrape.config import get_app_config, resolve_path
from aeroscrape.scraping.scraper import get_all_flights_data
from aeroscrape.storage.database import Database

logger = logging.getLogger(__name__)


def configure_logging() -> None:
    log_dir = resolve_path("logs")
    os.makedirs(log_dir, exist_ok=True)
    log_filename = os.path.join(log_dir, f"flight_ops_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s",
        handlers=[logging.FileHandler(log_filename), logging.StreamHandler()],
    )


def run_data_operations() -> int:
    """Run the requested operation and return a process exit code."""
    try:
        get_app_config()
        logger.info("Application configuration successfully accessed.")
    except Exception as e:
        logger.critical(f"Fatal error during application startup due to configuration issues: {e}")
        return 1

    parser = argparse.ArgumentParser(description="Flight data scraper, importer, and exporter.")
    parser.add_argument(
        "--date",
        type=str,
        help="Base date for flights (YYYY-MM-DD format) for CSV import or scraping. Defaults to current date if not provided.",
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--csv", type=str, help="Path to CSV file for import. Uses --date or current date if times in CSV are relative."
    )
    group.add_argument("--export-csv-path", type=str, help="Path to export the database content to a CSV file.")

    args = parser.parse_args()

    try:
        db = Database()
        if not db.test_connection():
            logger.error("Failed to connect to the database. Please check connection parameters and DB status.")
            return 1

        base_date = None
        if args.date:
            try:
                base_date = datetime.strptime(args.date, "%Y-%m-%d")
                logger.info(f"Using provided base date: {base_date.strftime('%Y-%m-%d')}")
            except ValueError:
                logger.error("Invalid date format for --date. Please use YYYY-MM-DD. Aborting.")
                return 1
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
                return 0
            logger.error(f"Failed to export database content to {args.export_csv_path}")
            return 1

        elif args.csv:
            logger.info(f"Attempting to import flights from CSV: {args.csv}")
            if not os.path.exists(args.csv):
                logger.error(f"CSV file not found at path: {args.csv}. Aborting import.")
                return 1
            result = db.insert_flights_from_csv(args.csv, base_date=base_date)

        else:  # Default behavior: scrape
            logger.info("Starting flight data scraping...")
            df = get_all_flights_data()
            if df is None or df.empty:
                logger.warning("No flight data was scraped. Nothing to import.")
                return 1
            logger.info(f"Successfully scraped {len(df)} flights.")
            result = db.insert_flights_from_dataframe(df, base_date=base_date)

    except Exception as e:
        logger.error(f"An unexpected error occurred in CLI execution: {str(e)}", exc_info=True)
        return 1

    # Skipped rows are expected (e.g. a flight with no airline yet); rejected rows or an empty load are not
    if result.failed or result.inserted + result.unchanged == 0:
        logger.error("Flight load failed: %s", result.summary())
        return 1
    return 0


def main() -> None:
    configure_logging()
    sys.exit(run_data_operations())


if __name__ == "__main__":
    main()
