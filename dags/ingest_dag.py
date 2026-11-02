"""Scrape the FIDS pages, validate the rows and upsert them into Postgres.

Runs every 3 hours: each airport page shows a rolling window of about a day, so scraping
once a day misses flights and their final status. The upsert keeps the latest status and
actual time of each flight, so re-running is safe.
"""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

from aeroscrape.pipeline.ingest_tasks import task_cleanup, task_load, task_scrape, task_validate_scrape
from aeroscrape.pipeline.watcher import fail_run

with DAG(
    dag_id="ingest_flights",
    description="Scrape flight boards, validate and load into Postgres.",
    schedule="0 */3 * * *",
    start_date=datetime(2026, 11, 1),
    catchup=False,
    max_active_runs=1,
    default_args={"owner": "aeroscrape", "retries": 2, "retry_delay": timedelta(minutes=5)},
    tags=["aeroscrape", "ingest"],
) as dag:
    scrape = PythonOperator(task_id="scrape", python_callable=task_scrape)
    validate = PythonOperator(task_id="validate", python_callable=task_validate_scrape)
    load = PythonOperator(task_id="load", python_callable=task_load)
    cleanup = PythonOperator(task_id="cleanup", python_callable=task_cleanup, trigger_rule="all_done")
    watcher = PythonOperator(task_id="watcher", python_callable=fail_run, trigger_rule="one_failed")

    scrape >> validate >> load >> cleanup
    [scrape, validate, load] >> watcher
