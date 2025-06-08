## Commands

All commands are run via the `main.py` script.

### 1. Scrape and Import Flight Data

This is the default operation if no specific import/export arguments are provided. It will scrape flight data and insert/update it into the database.

* **Basic usage (uses current date for relative times if scraper relies on it):**
    ```bash
    python3 src/main.py
    ```

* **With a specific base date (for parsing relative times during import, if applicable after scraping):**
    ```bash
    python3 src/main.py --date YYYY-MM-DD
    ```
    Example:
    ```bash
    python3 src/main.py --date 2025-05-26
    ```

### 2. Import Flight Data from CSV

This command imports flight data from a specified CSV file into the database.

* **Usage:**
    ```bash
    python3 src/main.py --csv /path/to/your/file.csv [--date YYYY-MM-DD]
    ```
    * `--csv /path/to/your/file.csv`: Path to the input CSV file.
    * `--date YYYY-MM-DD` (Optional but recommended): Specifies the base date for parsing time strings in the CSV that might not have a full date or weekday. If your CSV contains time strings like "HH:MM" or "Weekday HH:MM", this date helps resolve them correctly.

    Example:
    ```bash
    python3 src/main.py --csv ./flights_to_import.csv --date 2025-05-26
    ```

### 3. Export Flight Data to CSV

This command exports all flight data (arrivals and departures) from the database to a specified CSV file.

* **Usage:**
    ```bash
    python3 src/main.py --export-csv-path /path/to/output/file.csv
    ```
    * `--export-csv-path /path/to/output/file.csv`: Path where the exported CSV file will be saved.

    Example:
    ```bash
    python3 src/main.py --export-csv-path ./flights_export.csv
    ```

## Logging

Log files are created in a `logs` directory in the project root. Each run generates a new log file named with the current timestamp (e.g., `logs/flight_ops_YYYYMMDD_HHMMSS.log`).
