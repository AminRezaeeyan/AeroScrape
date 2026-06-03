# AeroScrape: End-to-End Flight Delay Prediction MLOps Pipeline

AeroScrape is a comprehensive MLOps (Machine Learning Operations) project designed to scrape real-time flight data, store it, train machine learning models to predict flight delays, and serve these predictions via a RESTful API. The entire pipeline is orchestrated using Apache Airflow and leverages MLflow for experiment tracking and model management, all running in a modular and maintainable local environment.

![Airflow Tasks Screenshot](https://github.com/aminrezaeeyan/AeroScrape/blob/main/screenshots/airflow-tasks.png?raw=true)
_Screenshot of the Airflow DAG orchestrating the ML pipeline._


## Table of Contents

-   [Features](#features)
-   [Architecture Overview](#architecture-overview)
-   [Getting Started (Docker Compose - Recommended)](#getting-started-docker-compose---recommended)
-   [Getting Started (Local Manual Setup)](#getting-started-local-manual-setup)
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
    * **Model Registry:** Centralized management and versioning of trained models, with automated alias assignment (e.g., to "production").
* **FastAPI Inference API:** Serves real-time flight delay predictions from the trained and registered ML models.
* **Modular & Reproducible:** Clean project structure, dependency management via `requirements.txt`, and containerization-ready design.

## Architecture Overview

The AeroScrape project is built with a modular architecture, separating concerns into distinct services and components:

1.  **Data Ingestion (Scraper):** A Python script (`src/scraper.py`) that fetches flight data from `fids.airport.ir`. The `src/cli.py` script orchestrates data scraping, CSV import, and database export.
2.  **Database:** A PostgreSQL database (`db` service in Docker setup) for storing raw and processed flight information.
3.  **ML Pipeline (Scripts):** A collection of Python scripts (`scripts/`) that perform the core ML workflow steps, from data validation to model training.
4.  **Orchestration (Apache Airflow):** Airflow DAGs (`dags/`) define and schedule the execution of the ML pipeline scripts as a Directed Acyclic Graph.
5.  **MLflow Tracking Server:** An MLflow instance (`mlflow_server` in Docker setup) that serves as a centralized repository for logging ML experiments, tracking metrics, and managing model versions in the Model Registry.
6.  **Inference API (FastAPI):** A FastAPI application (`src/api_service.py`) that loads the best performing ML model from MLflow Model Registry and provides a RESTful endpoint for real-time predictions.
7.  **Utilities (`utils/`):** A dedicated module for shared utility functions, including a robust configuration loader (`utils/config.py`) that handles both environment variables (`.env`) and structured YAML configuration (`config.yaml`).

---

## Getting Started (Docker Compose - Recommended)

This is the easiest way to launch the entire MLOps environment. All components (Database, MLflow, Airflow Webserver, Airflow Scheduler, FastAPI, Scraper) are configured with aligned volumes to share models, metadata, and artifacts.

### Prerequisites
* **Docker** and **Docker Compose** installed on your host system.

### Steps to Run

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/aminrezaeeyan/AeroScrape.git
   cd AeroScrape
   ```

2. **Configure Environment Variables:**
   Copy the template env file and specify your database credentials:
   ```bash
   cp template.env .env
   ```
   Modify `.env` to configure your database variables if needed. The defaults are already configured to connect internal services on the bridge network.

3. **Launch the Container Stack:**
   ```bash
   docker compose up -d
   ```
   This will build the required image, initialize the PostgreSQL schema, set up database tables, and start all services.

4. **Trigger the ML Pipeline:**
   * Open the Airflow Webserver UI in your browser at: **`http://localhost:8080`**
   * Log in using the default administrator credentials:
     * **Username:** `admin`
     * **Password:** `admin`
   * Locate the `flight_delay_prediction_pipeline` DAG, unpause it, and trigger a run.
   * Wait for all steps in the DAG (data validation, preprocessing, tuning, training, registry) to complete successfully and turn green.

5. **Load the Registered Model into the API:**
   Once the DAG finishes training and registers the model inside MLflow, trigger the hot-reload endpoint in your FastAPI service:
   ```bash
   curl -X POST http://localhost:8000/reload
   ```
   You can verify the API status by running:
   ```bash
   curl http://localhost:8000/health
   ```

---

## Getting Started (Local Manual Setup)

If you prefer to configure and run the services natively on your local machine instead of Docker, follow the manual steps below.

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

### Cloning the Repository

```bash
git clone https://github.com/aminrezaeeyan/AeroScrape.git
cd AeroScrape
```

### Virtual Environment Setup

```bash
python3.11 -m venv flight_env
source flight_env/bin/activate
```

### Configuration

1.  **Create `.env` file:**
    ```bash
    cp template.env .env
    ```
    Edit `.env` and replace `YOUR_ACTUAL_DB_PASSWORD_HERE` with a strong password for your local PostgreSQL database.

2.  **`config.yaml`:**
    Ensure this file exists in your project root with your pipeline parameters.

### Database Initialization

1.  **Connect to local PostgreSQL:**
    ```bash
    sudo -u postgres psql
    ```
2.  **Create Database and User:**
    ```sql
    CREATE DATABASE "flight-data";
    CREATE USER "flight-database-user" WITH ENCRYPTED PASSWORD 'your_secure_password_here'; -- Matches your .env
    GRANT ALL PRIVILEGES ON DATABASE "flight-data" TO "flight-database-user";
    \q
    ```
3.  **Initialize Database Schema:**
    ```bash
    psql -U flight-database-user -d flight-data -f init/init_db.sql
    ```

### Install Python Dependencies

```bash
pip install -r requirements.txt
```

### Running MLflow Tracking Server

1.  **Open a new terminal window, navigate to project root, and activate environment:**
    ```bash
    source flight_env/bin/activate
    ```
2.  **Start MLflow Server:**
    ```bash
    mkdir -p mlruns 
    mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlruns/mlruns.db --default-artifact-root ./mlruns
    ```
    *Access the MLflow UI at: `http://127.0.0.1:5000`*

### Running Apache Airflow

1.  **Setup Environment variables and folders:**
    ```bash
    export AIRFLOW_HOME=~/airflow
    mkdir -p "$AIRFLOW_HOME"
    rm -rf "$AIRFLOW_HOME/dags"
    ln -s "$(pwd)/dags" "$AIRFLOW_HOME/dags"
    ln -s "$(pwd)/scripts" "$AIRFLOW_HOME/scripts"
    ln -s "$(pwd)/utils" "$AIRFLOW_HOME/utils"
    ln -s "$(pwd)/config.yaml" "$AIRFLOW_HOME/config.yaml"
    ln -s "$(pwd)/data" "$AIRFLOW_HOME/data"
    ln -s "$(pwd)/models" "$AIRFLOW_HOME/models"
    ln -s "$(pwd)/mlruns" "$AIRFLOW_HOME/mlruns"
    ```
2.  **Initialize Metadata Database:**
    ```bash
    airflow db migrate
    ```
3.  **Create Admin User:**
    ```bash
    airflow users create \
        --username admin \
        --firstname Admin \
        --lastname User \
        --role Admin \
        --email admin@example.com
    ```
4.  **Configure Airflow to load `.env`:**
    Add your path inside `[core]` under `$AIRFLOW_HOME/airflow.cfg`:
    ```ini
    [core]
    env_file = /home/amin/AeroScrape/.env
    ```
5.  **Start Webserver and Scheduler (Keep terminals running):**
    ```bash
    airflow webserver -p 8080
    # In another terminal:
    airflow scheduler
    ```

### Running FastAPI Inference Service

```bash
uvicorn src.api_service:app --host 0.0.0.0 --port 8000 --reload
```
*Access docs at: `http://127.0.0.1:8000/docs`*

---

## Usage Workflow

1.  **Data Ingestion:**
    * Run the scraper to collect fresh flight data:
        ```bash
        python3 src/cli.py
        ```
2.  **ML Pipeline Execution:**
    * Trigger the Airflow DAG (`http://localhost:8080`).
3.  **Model Management:**
    * Access MLflow UI (`http://127.0.0.1:5000`) and ensure your model versions are promoted to stage/aliased to match your environment configs.
4.  **Real-time Prediction:**
    * Send prediction payloads to your FastAPI server (`http://localhost:8000/predict`).

---

## Data Analysis with Apache Superset

![Apache Superset Dashboard](https://github.com/aminrezaeeyan/AeroScrape/blob/main/screenshots/apache-superset.jpg?raw=true)
_A sample dashboard in Apache Superset for data analysis._

Apache Superset is an open-source data visualization and data exploration platform. In this project, we have utilized Superset for analyzing the scraped and processed flight data. It provides intuitive dashboards that allow for deep insights into the dataset, helping to understand flight patterns, delays, and other key metrics.

---

## MLflow Integration

![MLflow Metrics Dashboard](https://github.com/aminrezaeeyan/AeroScrape/blob/main/screenshots/mlflow-metrics.png?raw=true)
_Screenshot of MLflow UI showing metrics for different model runs._

MLflow is deeply integrated into the pipeline to ensure robust MLOps practices:

* **Experiment Tracking:** Every run of the `task_tune_hyperparameters`, `task_train_evaluate_regression`, and `task_train_evaluate_classification` tasks in Airflow logs its parameters, metrics, and artifacts (like the `config.yaml`, preprocessor, and best parameters JSON) to the MLflow Tracking Server.
* **Model Registry:** Trained models (`lgbm_regressor`, `logistic_classifier`) are automatically registered with the MLflow Model Registry. This enables versioning, stage/alias management, and a centralized repository for deployed models.

---

## CLI Tool (`src/cli.py`)

The `src/cli.py` file provides a command-line interface for direct interaction with the project's data scraping and database operations.

```bash
python3 src/cli.py --help
```

**Examples:**

* **Scrape and Import Flight Data (Default):**
    ```bash
    python3 src/cli.py --date 2026-06-03
    ```
* **Import Flight Data from CSV:**
    ```bash
    python3 src/cli.py --csv data/raw/flights_to_import.csv --date 2026-06-03
    ```
* **Export Flight Data to CSV:**
    ```bash
    python3 src/cli.py --export-csv-path data/exported_flights.csv
    ```

---

## FastAPI Endpoints

The FastAPI service provides a RESTful API for real-time flight delay predictions.

* **API Documentation:** `http://localhost:8000/docs` (Swagger UI)
* **Health Check:** `http://localhost:8000/health`

### `/predict` (POST)

**Usage:** Predicts flight delay based on input flight details.

**Request Body Example:**

```bash
curl -X 'POST' \
  'http://localhost:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "airline": "کاسپین",
    "airport": "فرودگاه مهرآباد",
    "destination_or_origin": "مشهد",
    "aircraft": "MD83",
    "scheduled_datetime": "2026-06-03 14:30:00"
  }'
```

**Response Body Example (Success):**

```json
{
  "predicted_delay_minutes": 28.97,
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