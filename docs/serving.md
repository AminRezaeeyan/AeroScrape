# Serving: one pipeline for training and prediction

The API predicts with the same model artifact and the same feature code that training used. Two things guarantee that: one registered MLflow model, and one feature function.

## One model artifact

`flight-delay-model` is an MLflow pyfunc ([`models/delay_model.py`](../src/aeroscrape/models/delay_model.py)) holding the quantile regressors, the classifier, its calibrator, the conformal margin and the decision threshold. It is logged with an input signature and example, so MLflow rejects input with missing columns or wrong types. The API loads it by alias (`models:/flight-delay-model@production`); nothing is read from local disk. Earlier versions loaded the model from the registry and its preprocessor from `models/preprocessor.joblib`, and the two could silently drift apart.

## One feature function

Training calls `build_features(departures, all_flights)`. For a request, [`serving/feature_lookup.py`](../src/aeroscrape/serving/feature_lookup.py) does the same for one departure:

1. Fetch from Postgres the flights its features look at: domestic departures from the 7 days before its date (rates), and the same day plus the 12 hours before it (rotation, live delays). `Database.fetch_flights` returns them in the raw scraped format.
2. Clean them with the training cleaner, `clean_flights`.
3. If the request names a scraped flight (`flight_number`), use its scraped row, registration included, exactly as training would.
4. Call `build_features(target, context)`.

`tests/test_parity.py` builds features for 50 departures through the training path and through the lookup and asserts they are identical, for both model variants. On the full 2025 dataset the same check passed for 80 departures with no differences.

### A small feature store

Earlier days change rarely, so the cleaned history is cached per date for 15 minutes; the flight's own day is fetched on every request so live delays are current. Measured with the stack on a laptop: about 310 ms per prediction (fetch, clean, features, model and SHAP), about 600 ms for the first request of a day, which fills the cache.

The alternative is a `feature_snapshot` table that Airflow refreshes with precomputed rates and recent arrivals. It would cut request latency, but it is a second copy of the feature logic's output that can go stale or disagree with training. At about 600 flights a day, computing on request is simpler and fast enough; the snapshot becomes worth it if volume or latency requirements grow.

### When information is missing

- A flight that wasn't scraped, or has no registration, gets no rotation or live features; the response lists them in `warnings` and the model's imputer fills them.
- If the database is unreachable, the API falls back to the request's own fields (calendar and identity) and says so in `warnings`. `/health` reports `feature_lookup`.
- The API connects as `aeroscrape_reader`, a role that can only read the flight tables.

## Lineage

Every training run records what it was built from, as MLflow run tags, as tags on the registered model version, in the model's metadata and in the report:

| Tag | Example |
| --- | --- |
| `git_commit`, `git_dirty` | `4a692b13...`, `false` |
| `raw_data_sha256` | hash of the raw CSV |
| `training_data_sha256` | hash of the exact training frame (features and targets) |
| `training_from`, `training_to`, `training_rows`, `test_rows` | 2025-05-22 to 2025-06-09, 2887, 687 |
| `features` | every feature column |
| `airflow_run_id` | the DAG run that trained it |

Containers have no `.git` directory, so `make train` passes the commit and dirty state in the DAG run's conf. A run started from the Airflow UI records `unknown`.

## Run directories

Each DAG run writes its intermediate files (cleaned flights, Parquet splits, tuned parameters, report) to `data/runs/<run_id>/` and passes only paths through XCom. Concurrent runs can't overwrite each other, and the cleanup task removes the directory when the run ends.
