from datetime import datetime, timedelta

from airflow import DAG
from airflow.models.param import Param
from airflow.operators.python import PythonOperator

from aeroscrape.pipeline.tasks import (
    task_clean_and_engineer_data,
    task_cleanup_processed_data,
    task_run_association_mining,
    task_split_data,
    task_train_and_evaluate,
    task_tune_hyperparameters,
    task_validate_raw_data,
)

# Define default arguments for the DAG
default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
    "start_date": datetime(2024, 6, 1),  # A fixed start date in the past
}

# Define the DAG
with DAG(
    dag_id="flight_delay_prediction_pipeline",
    default_args=default_args,
    description="Full MLOps pipeline with validation, cleaning, tuning, training, and evaluation.",
    schedule=None,
    catchup=False,
    tags=["ml", "flight_delay", "production"],
    params={
        "raw_data_path_override": Param(
            None,
            type=["null", "string"],
            title="Raw Data Path Override",
            description="Optional override for the raw data path (relative to project root). If None, uses path from config.yaml.",
        )
    },
) as dag:
    # Task 1: Validate the raw input data
    validate_op = PythonOperator(
        task_id="validate_raw_data_task",
        python_callable=task_validate_raw_data,
    )

    # Task 2: Clean raw data and engineer features
    clean_data_op = PythonOperator(
        task_id="clean_and_engineer_data_task",
        python_callable=task_clean_and_engineer_data,
    )

    # Task 3: Split by scheduled time into train, validation and test
    split_op = PythonOperator(
        task_id="split_data_task",
        python_callable=task_split_data,
    )

    # Task 4: Tune hyperparameters with rolling-origin CV on train+val (never test)
    tune_op = PythonOperator(
        task_id="tune_hyperparameters_task",
        python_callable=task_tune_hyperparameters,
    )

    # Task 5: Train both models, compare them with baselines on test, register them
    train_evaluate_op = PythonOperator(
        task_id="train_and_evaluate_task",
        python_callable=task_train_and_evaluate,
    )

    # Task 6: Run association mining
    association_mining_op = PythonOperator(
        task_id="run_association_task",
        python_callable=task_run_association_mining,
    )

    # Task 7: Clean up intermediate data files
    cleanup_op = PythonOperator(
        task_id="cleanup_processed_data_task",
        python_callable=task_cleanup_processed_data,
        trigger_rule="all_done",
    )

    validate_op >> clean_data_op >> split_op >> tune_op >> train_evaluate_op
    clean_data_op >> association_mining_op
    [train_evaluate_op, association_mining_op] >> cleanup_op
