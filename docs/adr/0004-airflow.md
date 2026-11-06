# 4. Orchestrate with Airflow, one DAG per cadence

**Status:** accepted, November 2026

## Context

Scraping, training and monitoring run on different schedules and fail for different reasons.

## Decision

Three Airflow DAGs: `ingest_flights` every 3 hours, `train_delay_model` weekly or when monitoring triggers it, `monitor_model` daily. Each ends with a watcher task so a failed task fails the run even though cleanup still runs. Task code lives in the package (`aeroscrape.pipeline`); the DAG files only wire it.

## Alternatives

- **Cron:** no retries, no run history, no dependencies between steps, nowhere to see which step failed.
- **One DAG for everything:** a site outage would block training, and a training failure would stop data collection.
- **Prefect or Dagster:** fine choices; Airflow was already in place and is widely known.

## Consequences

Airflow is the heaviest image (1.3 GB) and needs its own database. Each DAG can be triggered by hand (`make ingest`, `make train`, `make monitor`).
