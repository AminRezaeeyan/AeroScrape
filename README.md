# AeroScrape: End-to-End Flight Delay Prediction MLOps Pipeline

AeroScrape is a comprehensive MLOps (Machine Learning Operations) project designed to scrape real-time flight data, store it, train machine learning models to predict flight delays, and serve these predictions via a RESTful API. The entire pipeline is orchestrated using Apache Airflow, tracks experiment iterations via MLflow, and provides real-time explanations for predictions using SHAP (Explainable AI), served on a clean Streamlit interface.

![Airflow Tasks Screenshot](docs/screenshots/airflow-tasks.png)
_Screenshot of the Airflow DAG orchestrating the ML pipeline._


## Results

For each departure the model returns a **calibrated probability of a delay over 15 minutes** and a **likely delay range** (10th to 90th percentile, which contains about 8 in 10 flights). Two variants: one usable when the flight is booked, one an hour before departure, when the aircraft's previous delay is known.

Evaluated on a chronological split: model choices from rolling-origin CV on 2025-05-22 to 06-09, tested once on 06-10 to 06-12 (687 domestic departures, 26 airports). Changes are paired 95% bootstrap intervals against the best baseline (airline delay rate; airline × time-of-day median or percentiles). Full report in [docs/results/evaluation.md](docs/results/evaluation.md), regenerated with `make evaluate`.

| Model | MAE (min) | Change vs baseline | Range coverage (target 80%) | Brier | Change vs baseline |
| --- | --- | --- | --- | --- | --- |
| Best baseline | 24.7 | | 76.4% | 0.221 | |
| At booking | 24.0 | -0.7 [-1.2, -0.3] | 82.7% | 0.207 | -0.014 [-0.023, -0.005] |
| 1 h before departure | 23.0 | -1.7 [-2.6, -0.9] | 85.3% | 0.201 | -0.020 [-0.030, -0.011] |

Both variants beat the baselines on every metric, by modest margins. Most of the gain comes from recovering data the original cleaner discarded (training rows grew from 2,313 to 3,574) and from delay propagation: when the same aircraft's previous delay is known, the 1-hour model's error drops from 25.0 to 18.9 minutes. Aircraft registrations are only published at Mehrabad, so that applies to about a third of flights. The "likely delayed" flag is set for 75% precision and reaches 80-85% on test.

![Reliability on test](docs/results/reliability.png)

How the outputs are produced, calibrated and checked, including error analysis by segment: [docs/modeling.md](docs/modeling.md). Features and the ablation: [docs/features.md](docs/features.md). The first version of this project reported better-looking numbers from a random split and tuned on the test set; with an honest evaluation and the original features, it did not beat these baselines.

## Table of Contents

-   [Results](#results)
-   [Features](#features)
-   [Architecture Overview](#architecture-overview)
-   [Getting Started (Docker Compose - Recommended)](#getting-started-docker-compose---recommended)
-   [Getting Started (Local Manual Setup)](#getting-started-local-manual-setup)
-   [Inference & Explainability Interface](#inference--explainability-interface)
-   [Usage Workflow](#usage-workflow)
-   [Data Analysis with Apache Superset](#data-analysis-with-apache-superset)
-   [MLflow Integration](#mlflow-integration)
-   [CLI Tool (`aeroscrape`)](#cli-tool-aeroscrape)
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
* **Explainable AI (SHAP):** Integrates individual feature contributions for every single prediction, identifying factors that decrease or increase flight delays.
* **Inference Dashboard (Streamlit):** An interactive demo to query predictions and visualize SHAP force contributions on a clean web UI.

## Architecture Overview

The AeroScrape project is built with a modular architecture, separating concerns into distinct services and components:

1.  **Data Ingestion (Scraper):** A Python script (`aeroscrape.scraping.scraper`) that fetches flight data from `fids.airport.ir`. The `aeroscrape` command (`aeroscrape.cli`) orchestrates data scraping, CSV import, and database export.
2.  **Database:** A PostgreSQL database (`db` service in Docker setup) for storing raw and processed flight information.
3.  **ML Pipeline:** The `aeroscrape.data`, `aeroscrape.features`, `aeroscrape.models` and `aeroscrape.pipeline` packages that perform the core ML workflow steps, from data validation to model training.
4.  **Orchestration (Apache Airflow):** Airflow DAGs (`dags/`) define and schedule the execution of the ML pipeline scripts as a Directed Acyclic Graph.
5.  **MLflow Tracking Server:** An MLflow instance (`mlflow_server` in Docker setup) that serves as a centralized repository for logging ML experiments, tracking metrics, and managing model versions in the Model Registry.
6.  **Inference API (FastAPI):** A FastAPI application (`aeroscrape.serving.api`) that loads the native ML model from MLflow Model Registry and provides a RESTful endpoint returning predictions along with real-time SHAP values.
7.  **Inference Dashboard (Streamlit):** A user-friendly web interface (`aeroscrape.demo.streamlit_app`) allowing users to input flight details and examine predicted delays alongside contribution charts.
8.  **Configuration (`aeroscrape.config`):** A configuration loader that handles both environment variables (`.env`) and structured YAML configuration (`config.yaml`).

---

## Getting Started (Docker Compose - Recommended)

This is the easiest way to launch the entire MLOps environment. All components (Database, MLflow, Airflow Webserver, Airflow Scheduler, FastAPI, Streamlit, Scraper) are configured with aligned volumes to share models, metadata, and artifacts.

### Prerequisites
* **Docker** and **Docker Compose** installed on your host system.

### Steps to Run

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/aminrezaeeyan/AeroScrape.git
   cd AeroScrape
   ```

2. **Configure Environment Variables:**
   ```bash
   cp template.env .env
   ```
   Edit `.env` and set every password. `DB_ADMIN_*` is the Postgres superuser that only sets up the database; the scraper uses the limited `DB_USER` role and Airflow uses `AIRFLOW_DB_USER`. If a value contains `$`, write it as `$$`, because Docker Compose expands variables in this file.

3. **Download the Dataset and Launch the Stack:**
   ```bash
   make data   # downloads data/raw/data.csv and verifies its checksum
   make up
   ```
   This builds the image, creates the flight and Airflow databases, and starts all services. Postgres only runs its init scripts on an empty volume, so changing these roles later needs `docker compose down -v`, which deletes the stored flight data.

4. **Trigger the ML Pipeline:**
   * Open the Airflow Webserver UI in your browser at: **`http://localhost:8080`**
   * Log in with the `AIRFLOW_ADMIN_USER` and `AIRFLOW_ADMIN_PASSWORD` values from `.env`.
   * Run `make train`, or locate the `flight_delay_prediction_pipeline` DAG, unpause it, and trigger a run.
   * Wait for all steps in the DAG (data validation, preprocessing, tuning, training, registry) to complete successfully and turn green.

5. **Load the Registered Model into the API:**
   Once the DAG finishes training and registers `flight-delay-model` in MLflow, load it into the API:
   ```bash
   curl -X POST http://localhost:8000/reload
   ```

6. **Access the Services:**
   * **Inference Web UI (Streamlit):** `http://localhost:8501`
   * **Inference REST API (FastAPI):** `http://localhost:8000` (API Docs at `/docs`)
   * **Orchestrator (Apache Airflow):** `http://localhost:8080`
   * **Experiment Tracking (MLflow Server):** `http://localhost:5000`

---

## Getting Started (Local Manual Setup)

If you prefer to configure and run the services natively on your local machine instead of Docker, follow the manual steps below.

### Prerequisites

* **Git:** For cloning the repository.
* **Python 3.12** and **[uv](https://docs.astral.sh/uv/)**: `uv` installs the locked dependencies.
* **PostgreSQL:** Your application's scraper and database module (`aeroscrape.storage.database`) are designed to connect to a PostgreSQL database. You'll need a local PostgreSQL server running.
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

### Environment Setup

```bash
make setup               # installs uv.lock into .venv and the pre-commit hooks
source .venv/bin/activate
make data                # downloads the raw dataset
```

`make lint` and `make test` run the checks. All targets are listed in the [Makefile](Makefile).

### Configuration

1.  **Create `.env` file:**
    ```bash
    cp template.env .env
    ```
    Edit `.env` and set a strong password for your local PostgreSQL database. Also run `export MLFLOW_TRACKING_URI=http://127.0.0.1:5000`, because `config.yaml` points at the Docker hostname.

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
    psql -U flight-database-user -d flight-data -f init/10_init_db.sql
    ```

### Running MLflow Tracking Server

1.  **Open a new terminal window, navigate to project root, and run `source .venv/bin/activate`.**
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
    export AEROSCRAPE_HOME="$(pwd)"
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
uvicorn aeroscrape.serving.api:app --host 0.0.0.0 --port 8000 --reload
```
*Access docs at: `http://127.0.0.1:8000/docs`*

### Running Streamlit Frontend Dashboard

```bash
streamlit run src/aeroscrape/demo/streamlit_app.py --server.port 8501 --server.address 0.0.0.0
```
*Access dashboard at: `http://localhost:8501`*

---

## Inference & Explainability Interface

The demo dashboard provides a comprehensive evaluation of flight delay predictions. Beyond returning a numeric estimate, the backend computes individual feature importance using **SHAP**. This exposes exactly what factors are contributing to or preventing delays.

### 🎥 Live Prediction Interface
The interface allows users to select flight features and view estimated delays instantly.

![Demo Predictor Dashboard](docs/screenshots/demo-main.png)

_Streamlit web form predicting a 35.22-minute delay._


### 📊 Feature Explainability Breakdown
Our pipeline maps the numerical and categorical components to illustrate positive and negative contributors behind the prediction.

![SHAP Explainability Results and Charts](docs/screenshots/shap-explainability.png)

_SHAP breakdown of factors increasing or reducing delays alongside an interactive feature impact bar chart._

---

## Usage Workflow

1.  **Data Ingestion:**
    * Run the scraper to collect fresh flight data:
        ```bash
        aeroscrape
        ```
2.  **ML Pipeline Execution:**
    * Trigger the Airflow DAG (`http://localhost:8080`).
3.  **Model Management:**
    * Access MLflow UI (`http://127.0.0.1:5000`) and ensure your model versions are promoted to stage/aliased to match your environment configs.
4.  **Real-time Prediction:**
    * Send prediction payloads to your FastAPI server (`http://localhost:8000/predict`) or use the interactive Streamlit UI (`http://localhost:8501`).

---

## Data Analysis with Apache Superset

![Apache Superset Dashboard](docs/screenshots/apache-superset.jpg)
_A sample dashboard in Apache Superset for data analysis._

Apache Superset is an open-source data visualization and data exploration platform. In this project, we have utilized Superset for analyzing the scraped and processed flight data. It provides intuitive dashboards that allow for deep insights into the dataset, helping to understand flight patterns, delays, and other key metrics.

---

## MLflow Integration

![MLflow Metrics Dashboard](docs/screenshots/mlflow-metrics.png)
_Screenshot of MLflow UI showing metrics for different model runs._

MLflow is deeply integrated into the pipeline to ensure robust MLOps practices:

* **Experiment Tracking:** Every run of the `task_tune_hyperparameters`, `task_train_evaluate_regression`, and `task_train_evaluate_classification` tasks in Airflow logs its parameters, metrics, and artifacts (like the `config.yaml`, preprocessor, and best parameters JSON) to the MLflow Tracking Server.
* **Model Registry:** Trained models (`lgbm_regressor`, `logistic_classifier`) are automatically registered with the MLflow Model Registry. This enables versioning, stage/alias management, and a centralized repository for deployed models.

---

## CLI Tool (`aeroscrape`)

The `aeroscrape` command provides a command-line interface for direct interaction with the project's data scraping and database operations.

```bash
aeroscrape --help
```

**Examples:**

* **Scrape and Import Flight Data (Default):**
    ```bash
    aeroscrape --date 2026-06-03
    ```
* **Import Flight Data from CSV:**
    ```bash
    aeroscrape --csv data/raw/flights_to_import.csv --date 2026-06-03
    ```
* **Export Flight Data to CSV:**
    ```bash
    aeroscrape --export-csv-path data/exported_flights.csv
    ```

---

## FastAPI Endpoints

The FastAPI service provides a RESTful API for real-time flight delay predictions.

* **API Documentation:** `http://localhost:8000/docs` (Swagger UI)
* **Health Check:** `http://localhost:8000/health`

### `/predict` (POST)

Returns a calibrated probability of a delay over 15 minutes, a likely delay range and the main factors. Features are built from the flight database with the same code as training ([docs/serving.md](docs/serving.md)); `flight_number` and `register` are optional, but with them the aircraft's rotation is used.

```bash
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "airline": "وارش",
    "airport": "فرودگاه مهرآباد",
    "destination_or_origin": "مشهد",
    "aircraft": "737",
    "scheduled_datetime": "2026-11-03 08:05:00",
    "flight_number": "VRH5840"
  }'
```

```json
{
  "model_version": "2",
  "p_delay_over_15": 0.664,
  "likely_delayed": false,
  "decision_threshold": 0.69,
  "delay_minutes": {"p10": 4.7, "p50": 20.1, "p90": 107.9},
  "top_factors": [
    {"feature": "nth_flight_of_day_for_aircraft", "value": "1", "contribution": 0.163},
    {"feature": "destination_or_origin", "value": "مشهد", "contribution": 0.136}
  ],
  "explanation_units": "log-odds of a delay over 15 minutes, from the uncalibrated LightGBM classifier",
  "warnings": ["Not known for this flight, filled by the model: scheduled_turnaround_min"],
  "predicted_delay_minutes": 20.1,
  "is_delayed": false
}
```

`predicted_delay_minutes` and `is_delayed` repeat `p50` and `likely_delayed` for clients of the earlier response format.

## Contributors

* **Mahan Zavari** (mahanzavari@gmail.com)
* **Amin Rezaeeyan** (rezaeeyanamin@gmail.com)

## Acknowledgements

This project was developed under the esteemed supervision of **Dr. Hamidreza Shahriari** in the **Amirkabir University Of Technology (AUT-CE)**.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
