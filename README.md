# AeroScrape: End-to-End Flight Delay Prediction MLOps Pipeline

AeroScrape is a comprehensive MLOps (Machine Learning Operations) project designed to scrape real-time flight data, store it, train machine learning models to predict flight delays, and serve these predictions via a RESTful API. The entire pipeline is orchestrated using Apache Airflow and leverages MLflow for experiment tracking and model management, all running in a modular and maintainable local environment.

![Airflow Tasks Screenshot](https://github.com/aminrezaeeyan/AeroScrape/blob/main/screenshots/airflow-tasks.png?raw=true)
_Screenshot of the Airflow DAG orchestrating the ML pipeline._


## Table of Contents

-   [Features](#features)
-   [Architecture Overview](#architecture-overview)
-   [Getting Started](#getting-started)
    -   [Prerequisites](#prerequisites)
    -   [Cloning the Repository](#cloning-the-repository)
    -   [Virtual Environment Setup](#virtual-environment-setup)
    -   [Configuration](#configuration)
    -   [Database Initialization](#database-initialization)
    -   [Install Python Dependencies](#install-python-dependencies)
    -   [Running MLflow Tracking Server](#running-mlflow-tracking-server)
    -   [Running Apache Airflow](#running-apache-airflow)
    -   [Running FastAPI Inference Service](#running-fastapi-inference-service)
-   [Usage Workflow](#usage-workflow)
-   [Data Analysis with Apache Superset](#data-analysis-with-apache-superset)
-   [MLflow Integration](#mlflow-integration)
-   [CLI Tool (`src/cli.py`)](#cli-tool-srccli.py)
-   [FastAPI Endpoints](#fastapi-endpoints)
-   [Contributors](#contributors)
-   [Acknowledgements](#acknowledgements)
-   [License](#license)

## Features

* **Flight Data Scraping:** Real-time collection of flight arrival and departure data from `fids.airport.ir`.
* **Data Storage:** Persistent storage of scraped and processed flight data in a PostgreSQL database.
* **Automated ML Pipeline (Airflow):**
    * Raw data validation using `pandera`.
    * Data cleaning and feature engineering (e.g., time-of-day, season, holiday indicators).
    * Data preprocessing (splitting, scaling, one-hot encoding).
    * Hyperparameter tuning for regression (LightGBM) and classification (Logistic Regression) models using Optuna.
    * Model training and evaluation.
    * Cleanup of intermediate data.
* **MLflow Integration:**
    * **Experiment Tracking:** Logs all model training parameters, metrics, and artifacts.
    * **Model Registry:** Centralized management and versioning of trained models, with stage transitions (e.g., to "Production").
* **FastAPI Inference API:** Serves real-time flight delay predictions from the trained and registered ML models.
* **Modular & Reproducible:** Clean project structure, dependency management via `requirements.txt`, and containerization-ready design (with provided Docker setup).

## Architecture Overview

The AeroScrape project is built with a modular architecture, separating concerns into distinct services and components:

1.  **Data Ingestion (Scraper):** A Python script (`src/scraper.py`) that fetches flight data from `fids.airport.ir`. The `src/cli.py` script orchestrates data scraping, CSV import, and database export.
2.  **Database:** A PostgreSQL database (`db` service in Docker setup) for storing raw and processed flight information.
3.  **ML Pipeline (Scripts):** A collection of Python scripts (`scripts/`) that perform the core ML workflow steps, from data validation to model training.
4.  **Orchestration (Apache Airflow):** Airflow DAGs (`dags/`) define and schedule the execution of the ML pipeline scripts as a Directed Acyclic Graph.
5.  **MLflow Tracking Server:** An MLflow instance (`mlflow_server` in Docker setup) that serves as a centralized repository for logging ML experiments, tracking metrics, and managing model versions in the Model Registry.
6.  **Inference API (FastAPI):** A FastAPI application (`src/api_service.py`) that loads the best performing ML model from MLflow Model Registry and provides a RESTful endpoint for real-time predictions.
7.  **Utilities (`utils/`):** A dedicated module for shared utility functions, including a robust configuration loader (`utils/config.py`) that handles both environment variables (`.env`) and structured YAML configuration (`config.yaml`).

## Getting Started

Follow these steps to set up and run the AeroScrape project on your local machine.

### Prerequisites

* **Git:** For cloning the repository.
* **Python 3.11:** It's crucial to use Python 3.11 as Airflow 2.x versions have specific Python compatibility.
* **`python3.11-venv`:** For creating isolated Python environments.
    ```bash
    sudo apt update
    sudo apt install python3.11 python3.11-venv
    ```
* **PostgreSQL:** Your application's scraper and database module (`src/database.py`) are designed to connect to a PostgreSQL database. You'll need a local PostgreSQL server running.
    ```bash
    sudo apt install postgresql postgresql-contrib
    sudo systemctl start postgresql
    sudo systemctl enable postgresql
    ```
    You'll also need to create a database and user for your application (e.g., `flight-data` database, `flight-database-user` user).

### Cloning the Repository

```bash
git clone https://github.com/aminrezaeeyan/AeroScrape.git
cd AeroScrape
```
### Virtual Environment Setup

It's highly recommended to use a virtual environment to manage project dependencies.

```bash
python3.11 -m venv flight_env
source flight_env/bin/activate
```

### Configuration

The project uses a combination of `.env` for sensitive secrets and `config.yaml` for structured, non-sensitive settings.

1.  **Create `.env` file:**
    Copy the `template.env` file and rename it to `.env`.
    ```bash
    cp template.env .env
    ```
    **Edit `.env`** and replace `YOUR_ACTUAL_DB_PASSWORD_HERE` with a strong password for your PostgreSQL database user.

2.  **`config.yaml`:**
    This file contains all other project configurations. It should already be in your repository. Ensure it's up-to-date with the latest structure provided in the project.

### Database Initialization

Your application's scraper and ML pipeline will connect to a PostgreSQL database.

1.  **Connect to PostgreSQL:**
    ```bash
    sudo -u postgres psql
    ```
2.  **Create Database and User (if you haven't already):**
    ```sql
    CREATE DATABASE "flight-data";
    CREATE USER "flight-database-user" WITH ENCRYPTED PASSWORD 'your_secure_password_here'; -- Use the same password as in .env
    GRANT ALL PRIVILEGES ON DATABASE "flight-data" TO "flight-database-user";
    \q
    ```
3.  **Initialize Database Schema:**
    Apply the SQL schema from `init/init_db.sql`.
    ```bash
    psql -U flight-database-user -d flight-data -f init/init_db.sql
    ```

### Install Python Dependencies

With your virtual environment active, install all required packages.

```bash
pip install -r requirements.txt
```

### Running MLflow Tracking Server

The MLflow server provides the UI and API for tracking experiments and managing models.

1.  **Open a new terminal window.**
2.  **Navigate to your project root:** `cd ~/AeroScrape`
3.  **Activate your virtual environment:** `source flight_env/bin/activate`
4.  **Start MLflow Server:**
    ```bash
    mkdir -p mlruns # Ensure mlruns directory exists for local storage
    mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlruns/mlruns.db --default-artifact-root ./mlruns
    ```
    *Keep this terminal window open and running.*
    *Access the MLflow UI at: `http://127.0.0.1:5000`*

### Running Apache Airflow

Airflow orchestrates your ML pipeline. We'll set it up using its simplest local mode.

1.  **Perform One-Time Airflow Setup (if you haven't already):**
    * **Open a new terminal window.**
    * **Navigate to your project root:** `cd ~/AeroScrape`
    * **Activate your virtual environment:** `source flight_env/bin/activate`
    * **Set `AIRFLOW_HOME`:** This tells Airflow where to put its config, DB, and logs.
        ```bash
        export AIRFLOW_HOME=~/airflow
        ```
        *Consider adding this to your `~/.bashrc` or `~/.zshrc` for convenience.*
    * **Create `AIRFLOW_HOME` and Symlink Project Folders:**
        ```bash
        mkdir -p "$AIRFLOW_HOME"
        rm -rf "$AIRFLOW_HOME/dags" # Remove default dags if any
        ln -s "$(pwd)/dags" "$AIRFLOW_HOME/dags"
        ln -s "$(pwd)/scripts" "$AIRFLOW_HOME/scripts"
        ln -s "$(pwd)/utils" "$AIRFLOW_HOME/utils"
        ln -s "$(pwd)/config.yaml" "$AIRFLOW_HOME/config.yaml"
        ln -s "$(pwd)/data" "$AIRFLOW_HOME/data"
        ln -s "$(pwd)/models" "$AIRFLOW_HOME/models"
        ln -s "$(pwd)/mlruns" "$AIRFLOW_HOME/mlruns"
        ```
    * **Initialize Airflow's Metadata Database:** (This creates `airflow.db` in `$AIRFLOW_HOME`)
        ```bash
        airflow db migrate
        ```
    * **Create an Airflow Admin User:** (For logging into the Airflow UI)
        ```bash
        airflow users create \
            --username admin \
            --firstname Admin \
            --lastname User \
            --role Admin \
            --email admin@example.com
        # Set a strong password when prompted!
        ```
    * **Configure Airflow to load `.env`:**
        Edit `$AIRFLOW_HOME/airflow.cfg` using `nano "$AIRFLOW_HOME/airflow.cfg"` and add/modify the `env_file` line in the `[core]` section:
        ```ini
        # In $AIRFLOW_HOME/airflow.cfg
        [core]
        env_file = /home/amin/AeroScrape/.env # Ensure this path is correct for your project
        ```

2.  **Start Airflow Webserver:**
    * **Open a new terminal window.**
    * **Navigate to your project root:** `cd ~/AeroScrape`
    * **Activate your virtual environment:** `source flight_env/bin/activate`
    * **Ensure `AIRFLOW_HOME` is set:** `export AIRFLOW_HOME=~/airflow`
    * ```bash
        airflow webserver -p 8080
        ```
    * *Keep this terminal window open and running.*
    * *Access the Airflow UI at: `http://localhost:8080`*

3.  **Start Airflow Scheduler:**
    * **Open a new terminal window.**
    * **Navigate to your project root:** `cd ~/AeroScrape`
    * **Activate your virtual environment:** `source flight_env/bin/activate`
    * **Ensure `AIRFLOW_HOME` is set:** `export AIRFLOW_HOME=~/airflow`
    * ```bash
        airflow scheduler
        ```
    * *Keep this terminal window open and running.*

### Running FastAPI Inference Service

Once your Airflow pipeline has successfully trained and registered a model (and you've transitioned it to "Production" stage in MLflow UI), you can run the API.

1.  **Open a new terminal window.**
2.  **Navigate to your project root:** `cd ~/AeroScrape`
3.  **Activate your virtual environment:** `source flight_env/bin/activate`
4.  **Run FastAPI service:**
    ```bash
    uvicorn src.api_service:app --host 0.0.0.0 --port 8000 --reload
    ```
    *Keep this terminal window open and running.*
    *Access the API documentation at: `http://127.0.0.1:8000/docs`*
    *Access the health check at: `http://127.0.0.1:8000/health`*

## Usage Workflow

1.  **Data Ingestion:**
    * Run the scraper to collect fresh flight data from `fids.airport.ir` and populate the database:
        ```bash
        python3 src/cli.py
        ```
    * (Optional) Import data from a CSV:
        ```bash
        python3 src/cli.py --csv data/raw/your_data.csv --date 2025-07-05
        ```
    * (Optional) Export data from the database to CSV:
        ```bash
        python3 src/cli.py --export-csv-path data/exported_flights.csv
        ```
2.  **ML Pipeline Execution:**
    * Access the Airflow UI (`http://localhost:8080`).
    * Enable the `flight_delay_prediction_pipeline` DAG.
    * Trigger a new DAG run.
    * Monitor task progress in the Airflow UI.
3.  **Model Management:**
    * Access the MLflow UI (`http://127.0.0.1:5000`).
    * Review experiment runs, parameters, and metrics.
    * Navigate to the "Models" section to view registered models (`flight-delay-regressor`, `flight-delay-classifier`).
    * **Crucially:** Ensure the desired model version (e.g., the one from your latest successful Airflow run) is transitioned to the "Production" stage for API serving. You can do this by clicking on the model version and using the "Stage" dropdown.
4.  **Real-time Prediction:**
    * Once the FastAPI service is running, send prediction requests to `http://127.0.0.1:8000/predict` using the format described in `docs/request_style.txt`.

## Data Analysis with Apache Superset

![Apache Superset Dashboard](https://github.com/aminrezaeeyan/AeroScrape/blob/main/screenshots/apache-superset.jpg?raw=true)
_A sample dashboard in Apache Superset for data analysis._

Apache Superset is an open-source data visualization and data exploration platform. In this project, we have utilized Superset for analyzing the scraped and processed flight data. It provides powerful, intuitive dashboards that allow for deep insights into the dataset, helping to understand flight patterns, delays, and other key metrics.

**Tip:** You can easily connect your PostgreSQL database (the `flight-data` database) to Apache Superset. This allows you to build custom dashboards and charts directly from your stored data, providing a dynamic and interactive way to explore the information.

## MLflow Integration

![MLflow Metrics Dashboard](https://github.com/aminrezaeeyan/AeroScrape/blob/main/screenshots/mlflow-metrics.png?raw=true)
_Screenshot of MLflow UI showing metrics for different model runs._

MLflow is deeply integrated into the pipeline to ensure robust MLOps practices:

* **Experiment Tracking:** Every run of the `task_tune_hyperparameters`, `task_train_evaluate_regression`, and `task_train_evaluate_classification` tasks in Airflow logs its parameters, metrics, and artifacts (like the `config.yaml`, preprocessor, and best parameters JSON) to the MLflow Tracking Server. This provides a detailed history of all experiments.
* **Model Registry:** Trained models (`lgbm_regressor`, `logistic_classifier`) are automatically registered with the MLflow Model Registry. This enables versioning, stage management (e.g., promoting models to "Production"), and a centralized repository for deployed models. The FastAPI service then loads the "Production" stage of the `flight-delay-regressor` model, ensuring the API always serves the currently approved model version.

## CLI Tool (`src/cli.py`)

The `src/cli.py` file provides a command-line interface for direct interaction with the project's data scraping and database operations. This is useful for manual data ingestion, debugging, or one-off tasks outside of the Airflow pipeline.

**Usage:**

```bash
python3 src/cli.py --help
```

**Help Output:**

```
usage: cli.py [-h] [--date DATE] [--csv CSV | --export-csv-path EXPORT_CSV_PATH]

Flight data scraper, importer, and exporter.

options:
  -h, --help            show this help message and exit
  --date DATE           Base date for flights (YYYY-MM-DD format) for CSV import or scraping. Defaults to current date if not provided.

mutually exclusive arguments:
  --csv CSV             Path to CSV file for import. Uses --date or current date if times in CSV are relative.
  --export-csv-path EXPORT_CSV_PATH
                        Path to export the database content to a CSV file.
```

**Examples:**

* **Scrape and Import Flight Data (Default):**
    ```bash
    python3 src/cli.py
    ```
    (Uses current date as base date for time parsing)
    ```bash
    python3 src/cli.py --date 2025-07-05
    ```
    (Uses a specific base date for time parsing)

* **Import Flight Data from CSV:**
    ```bash
    python3 src/cli.py --csv data/raw/flights_to_import.csv --date 2025-07-05
    ```

* **Export Flight Data to CSV:**
    ```bash
    python3 src/cli.py --export-csv-path data/exported_flights.csv
    ```

## FastAPI Endpoints

The FastAPI service provides a RESTful API for real-time flight delay predictions.

* **API Documentation:** `http://127.0.0.1:8000/docs` (Swagger UI)
* **Health Check:** `http://127.0.0.1:8000/health`

### `/predict` (POST)

**Usage:** Predicts flight delay based on input flight details.

**Request Body Example:**

```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "airline": "ایران ایرتور",
  "airport": "بیرجند",
  "destination_or_origin": "بیرجند",
  "aircraft": "MD82",
  "scheduled_datetime": "2025-07-05 05:05:00"
}'
```

**Response Body Example (Success):**

```json
{
  "predicted_delay_minutes": 15.34,
  "is_delayed": true
}
```

## Contributors

* **Mahan Zavari** (mahanzavari@gmail.com)
* **Amin Rezaeeyan** (rezaeeyanamin@gmail.com)

## Acknowledgements

This project was developed under the esteemed supervision of **Dr. Hamidreza Shahriari** in the **Amirkabir University Of Technology (AUT-CE)**.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
