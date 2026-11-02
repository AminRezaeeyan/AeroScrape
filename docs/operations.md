# Operations: ingesting data, training, and promoting models

Two Airflow DAGs, a promotion gate between training and serving, and one-command rollback.

## The two DAGs

| DAG | Schedule | Tasks |
| --- | --- | --- |
| `ingest_flights` | every 3 hours | scrape → validate → load into Postgres → cleanup |
| `train_delay_model` | weekly | validate archive → build dataset → split → tune → train and evaluate → **gate** → promote or reject → cleanup |

They are separate because they run at different cadences and fail for different reasons: a site outage should not block training, and a training failure should not stop data collection.

- **Ingest every 3 hours.** Each airport page shows a rolling window of about a day, so a daily scrape misses flights and their final status. The load upserts on (flight number, scheduled time, airport), so a re-run updates rows instead of duplicating them.
- **Validation** fails a batch if a required field is missing or the page layout looks wrong, but tolerates a few flights without a status or airline (up to 5%); the loader skips those rows and reports how many. An earlier version rejected a whole 700-flight scrape because one flight had no status.
- **Training data** is the 2025 archive plus everything in Postgres (`data_sources` in `config.yaml`). The test window is the last 3 dates with data, so it moves forward as flights are scraped; with only one day of live data so far, it mixes two 2025 days with that day.
- Both DAGs end with a `watcher` task that fails the run if any task failed. Cleanup runs regardless, and without the watcher a failed run would show as successful.

## The promotion gate

Training registers the new model as `challenger`; it never touches `production` directly. `evaluate_gate` loads both versions from the registry, scores them on the same held-out flights, and promotes the challenger only if every check passes (thresholds under `pipeline.gate` in `config.yaml`):

| Check | Rule |
| --- | --- |
| Enough held-out flights | at least 200 |
| Better than the champion | the paired 95% bootstrap interval of the MAE change is entirely below zero |
| Better than the baseline | same, against the airline × time-of-day median |
| Calibration | Brier score at most 0.005 worse than the champion's |
| Range coverage | 70% to 90% of flights inside the 10-90% range |

With no champion yet, only the flight-count, baseline and coverage checks apply. If the champion was trained on data that overlaps the held-out window, the gate notes that the comparison favors the champion.

Each decision is an MLflow run named `promotion_gate` with the comparison table (`gate/comparison.md`), the metrics, the two versions, the decision and the action taken; the challenger's model version is tagged `gate_decision`.

### What it did on real runs

| Run | Challenger | Result | Why |
| --- | --- | --- | --- |
| Retrain, same data plus one live day | version 3 | rejected | MAE change vs champion +0.13 min [-0.10, +0.38]: no evidence it is better |
| Trained on 10% of the data (`make train TRAIN_FRACTION=0.1`) | version 4 | rejected | MAE +0.66 min [+0.27, +1.04] worse, Brier 0.232 vs 0.215 |

Version 2 kept serving both times. Rejecting a model that is merely not better is intended: replacing a model on noise adds risk without benefit.

## Rollback and hot swap

```bash
make model-status          # which versions are production and challenger
make rollback VERSION=3    # serve version 3
```

`rollback` moves the `production` alias and tags the version with the reason and the version it replaced. The API polls the registry every 2 minutes (`serving.registry_poll_seconds`) and, when the alias moves, loads the new version fully before swapping it in, so requests never see a half-loaded model; `POST /reload` checks immediately. In a test, the API switched to a rolled-back version 111 seconds after `make rollback`, with no restart. Every response and `/health` report the version being served.

## Commands

| Command | What it does |
| --- | --- |
| `make ingest` | scrape now (and unpause the 3-hourly schedule) |
| `make train` | train a challenger and run the gate; the commit is recorded for lineage |
| `make train TRAIN_FRACTION=0.1` | train on 10% of the data, to check that the gate rejects a worse model |
| `make model-status` | show the production and challenger versions |
| `make rollback VERSION=n` | serve version n |
