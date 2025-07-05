import os
import sys
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.models.param import Param
from datetime import datetime, timedelta

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

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
    import logging
    dag_logger = logging.getLogger(__name__)
    dag_logger.info("Successfully imported pipeline tasks for Airflow DAG.")
except ImportError as e:
    dag_logger.error(f"Error importing pipeline tasks for Airflow DAG: {e}")
    dag_logger.error(f"Please ensure project root ({project_root}) is correctly in sys.path: {sys.path}")
    raise

# Define default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=2),
    'start_date': datetime(2024, 6, 1), # A fixed start date in the past
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
            title="Raw Data Path Override",
            description="Optional override for the raw data path (relative to project root). If None, uses path from config.yaml."
        )
    }
) as dag:

    # Task 1: Validate the raw input data
    validate_op = PythonOperator(
        task_id='validate_raw_data_task',
        python_callable=task_validate_raw_data,
        provide_context=True, # Essential for ti (TaskInstance) access
    )

    # Task 2: Clean raw data and engineer features
    clean_data_op = PythonOperator(
        task_id='clean_and_engineer_data_task',
        python_callable=task_clean_and_engineer_data,
        provide_context=True,
    )

    # Task 3: Preprocess the cleaned data for modeling (split, scale, encode)
    preprocess_op = PythonOperator(
        task_id='preprocess_for_modeling_task',
        python_callable=task_preprocess_for_modeling,
        provide_context=True,
    )

    # Task 4: Tune hyperparameters for the models
    tune_op = PythonOperator(
        task_id='tune_hyperparameters_task',
        python_callable=task_tune_hyperparameters,
        provide_context=True,
    )

    # Task 5a: Train and evaluate the regression model
    train_eval_regression_op = PythonOperator(
        task_id='train_evaluate_regression_task',
        python_callable=task_train_evaluate_regression,
        provide_context=True,
    )

    # Task 5b: Train and evaluate the classification model
    train_eval_classification_op = PythonOperator(
        task_id='train_evaluate_classification_task',
        python_callable=task_train_evaluate_classification,
        provide_context=True,
    )

    # Task 6: Clean up intermediate data files
    cleanup_op = PythonOperator(
        task_id='cleanup_processed_data_task',
        python_callable=task_cleanup_processed_data,
        trigger_rule='all_done',
        provide_context=True,
    )

    # Define the task dependencies
    validate_op >> clean_data_op >> preprocess_op >> tune_op
    tune_op >> [train_eval_regression_op, train_eval_classification_op]
    [train_eval_regression_op, train_eval_classification_op] >> cleanup_op