from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.models.param import Param
from datetime import datetime, timedelta
import sys
import os

# Add the project root and scripts directory to Python's path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "scripts")
sys.path.insert(0, SCRIPTS_DIR)
sys.path.insert(0, PROJECT_ROOT)

# Import the task functions from your pipeline_tasks script
try:
    from scripts.pipeline_tasks import (
        task_validate_raw_data,
        task_clean_and_engineer_data,
        task_preprocess_for_modeling,
        task_tune_hyperparameters,
        task_train_evaluate_regression,
        task_train_evaluate_classification,
        task_cleanup_processed_data
    )
    print("Successfully imported pipeline tasks for Airflow DAG.")
except ImportError as e:
    print(f"Error importing pipeline tasks for Airflow DAG: {e}")
    print(f"Please ensure all required scripts exist in: {SCRIPTS_DIR}")
    raise

# Define default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=2),
    'start_date': datetime(2024, 6, 1),
}

# Define the DAG
with DAG(
    dag_id='flight_delay_prediction_pipeline',
    default_args=default_args,
    description='Full MLOps pipeline with validation, cleaning, tuning, training, and evaluation.',
    schedule=None,
    catchup=False,
    tags=['ml', 'flight_delay', 'production'],
    params={
        "raw_data_path_override": Param(
            None,
            type=["null", "string"],
            description="Optional override for the raw data path (relative to project root). If None, uses path from pipeline_config.yaml."
        )
    }
) as dag:

    # Task 1: Validate the raw input data
    validate_op = PythonOperator(
        task_id='validate_raw_data_task',
        python_callable=task_validate_raw_data,
    )

    # Task 2: Clean raw data and engineer features
    clean_data_op = PythonOperator(
        task_id='clean_and_engineer_data_task',
        python_callable=task_clean_and_engineer_data,
    )

    # Task 3: Preprocess the cleaned data for modeling (split, scale, encode)
    preprocess_op = PythonOperator(
        task_id='preprocess_for_modeling_task',
        python_callable=task_preprocess_for_modeling,
    )

    # Task 4: Tune hyperparameters for the models
    tune_op = PythonOperator(
        task_id='tune_hyperparameters_task',
        python_callable=task_tune_hyperparameters,
    )

    # Task 5a: Train and evaluate the regression model
    train_eval_regression_op = PythonOperator(
        task_id='train_evaluate_regression_task',
        python_callable=task_train_evaluate_regression,
    )

    # Task 5b: Train and evaluate the classification model
    train_eval_classification_op = PythonOperator(
        task_id='train_evaluate_classification_task',
        python_callable=task_train_evaluate_classification,
    )

    # Task 6: Clean up intermediate data files
    cleanup_op = PythonOperator(
        task_id='cleanup_processed_data_task',
        python_callable=task_cleanup_processed_data,
        trigger_rule='all_done',
    )

    # --- Define the full dependency chain ---
    validate_op >> clean_data_op >> preprocess_op >> tune_op
    tune_op >> [train_eval_regression_op, train_eval_classification_op]
    [train_eval_regression_op, train_eval_classification_op] >> cleanup_op