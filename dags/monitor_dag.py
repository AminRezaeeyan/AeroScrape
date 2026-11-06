"""Monitor the served model: actual outcomes, live performance, input drift, retraining.

Runs daily. Predictions logged by the API are matched to the delays the scraper collected,
live MAE, Brier and coverage are compared with the offline test numbers, and recent inputs
are compared with the training inputs. Summary numbers go to the Pushgateway, where
Prometheus alerts on them. A sustained degradation or drift triggers train_delay_model;
its promotion gate still decides whether the new model replaces the current one.
"""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator, ShortCircuitOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

from aeroscrape.pipeline.monitor_tasks import (
    MONITOR_DAG,
    TRAIN_DAG,
    decide_retrain,
    task_apply_retention,
    task_detect_drift,
    task_join_outcomes,
    task_live_performance,
)
from aeroscrape.pipeline.watcher import fail_run

with DAG(
    dag_id=MONITOR_DAG,
    description="Join predictions with actual delays, track live performance and input drift.",
    schedule="30 6 * * *",  # after the night's flights have been scraped
    start_date=datetime(2026, 11, 1),
    catchup=False,
    max_active_runs=1,
    default_args={"owner": "aeroscrape", "retries": 1, "retry_delay": timedelta(minutes=5)},
    tags=["aeroscrape", "monitoring"],
) as dag:
    outcomes = PythonOperator(task_id="join_outcomes", python_callable=task_join_outcomes)
    performance = PythonOperator(task_id="live_performance", python_callable=task_live_performance)
    drift = PythonOperator(task_id="detect_drift", python_callable=task_detect_drift)
    decide = ShortCircuitOperator(
        task_id="decide_retrain", python_callable=decide_retrain, ignore_downstream_trigger_rules=False
    )
    retrain = TriggerDagRunOperator(
        task_id="trigger_retraining",
        trigger_dag_id=TRAIN_DAG,
        conf={"triggered_by": MONITOR_DAG, "reason": "{{ ti.xcom_pull(task_ids='decide_retrain', key='reason') }}"},
    )
    retention = PythonOperator(task_id="apply_retention", python_callable=task_apply_retention)
    watcher = PythonOperator(task_id="watcher", python_callable=fail_run, trigger_rule="one_failed")

    outcomes >> performance >> decide >> retrain
    drift >> decide
    [performance, drift] >> retention
    [outcomes, performance, drift, decide, retrain, retention] >> watcher
