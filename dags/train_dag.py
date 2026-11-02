"""Build the dataset, train and evaluate a challenger, and promote it only if it passes the gate.

The new model is registered with the alias "challenger". The gate scores it and the current
"production" model on the same newest held-out flights; the branch then promotes it or
keeps the champion. Every decision is logged to MLflow as a "promotion_gate" run.
"""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.models.param import Param
from airflow.operators.python import BranchPythonOperator, PythonOperator

from aeroscrape.pipeline.tasks import (
    choose_branch,
    task_build_dataset,
    task_cleanup_processed_data,
    task_evaluate_gate,
    task_promote,
    task_reject,
    task_run_association_mining,
    task_split_data,
    task_train_and_evaluate,
    task_tune_hyperparameters,
    task_validate_raw_data,
)
from aeroscrape.pipeline.watcher import fail_run

with DAG(
    dag_id="train_delay_model",
    description="Train, evaluate and gate the flight delay model.",
    schedule="@weekly",
    start_date=datetime(2026, 11, 1),
    catchup=False,
    max_active_runs=1,
    default_args={"owner": "aeroscrape", "retries": 1, "retry_delay": timedelta(minutes=2)},
    tags=["aeroscrape", "training"],
    params={
        "raw_data_path_override": Param(
            None,
            type=["null", "string"],
            description="Train on this raw CSV (relative to the project root) instead of the configured sources.",
        ),
        "train_fraction": Param(
            1.0,
            type="number",
            minimum=0.01,
            maximum=1.0,
            description="Train on this share of train+val, e.g. 0.1 to check that the gate rejects a worse model.",
        ),
    },
) as dag:
    validate = PythonOperator(task_id="validate_archive", python_callable=task_validate_raw_data)
    build = PythonOperator(task_id="build_dataset", python_callable=task_build_dataset)
    split = PythonOperator(task_id="split_data", python_callable=task_split_data)
    tune = PythonOperator(task_id="tune_hyperparameters", python_callable=task_tune_hyperparameters)
    train = PythonOperator(task_id="train_and_evaluate", python_callable=task_train_and_evaluate)
    gate = PythonOperator(task_id="evaluate_gate", python_callable=task_evaluate_gate)
    branch = BranchPythonOperator(task_id="choose_branch", python_callable=choose_branch)
    promote = PythonOperator(task_id="promote", python_callable=task_promote)
    reject = PythonOperator(task_id="reject", python_callable=task_reject)
    mining = PythonOperator(task_id="association_mining", python_callable=task_run_association_mining)
    cleanup = PythonOperator(task_id="cleanup", python_callable=task_cleanup_processed_data, trigger_rule="all_done")
    watcher = PythonOperator(task_id="watcher", python_callable=fail_run, trigger_rule="one_failed")

    validate >> build >> split >> tune >> train >> gate >> branch >> [promote, reject]
    build >> mining
    [promote, reject, mining] >> cleanup
    [validate, build, split, tune, train, gate, branch, promote, reject, mining] >> watcher
