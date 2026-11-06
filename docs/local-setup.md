# Local setup and commands

Docker Compose (`make up`) is the supported way to run everything. This page covers working on the
code without Docker, and lists every `make` target.

## Development environment

Needs Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
make setup     # installs uv.lock into .venv (every extra) and the pre-commit hooks
make data      # the 2025 archive, checksum verified
make lint      # ruff, formatting and mypy
make test      # the test suite; database tests run only with AEROSCRAPE_TEST_DB=1
```

`make evaluate` tunes, trains and evaluates locally on the archive and writes
`reports/evaluation.md`; it needs no database or MLflow server. `SOURCE=bts make evaluate` does the
same on the US data after `make bts-data`.

## Running the services natively

1. **Postgres.** Create a database and run the init scripts in order, as the Postgres container
   does. They read the role names from the environment:

   ```bash
   export PGHOST=localhost POSTGRES_USER=postgres POSTGRES_DB=flights \
          APP_DB_USER=aeroscrape_app APP_DB_PASSWORD=... \
          READER_DB_USER=aeroscrape_reader READER_DB_PASSWORD=... \
          GRAFANA_DB_USER=aeroscrape_grafana GRAFANA_DB_PASSWORD=...
   createdb flights
   psql -d flights -f init/10_init_db.sql
   for s in init/20_*.sh init/30_*.sh init/40_*.sh; do bash "$s"; done
   psql -d flights -f init/50_add_flight_updated_at.sql
   ```

   Then set `DB_HOST=localhost`, `DB_PORT=5432` and the `DB_*` values of the role you run as in
   `.env`. The API uses the reader role, everything else the app role.

2. **MLflow.** `mlflow server --port 5000 --backend-store-uri sqlite:///mlruns/mlruns.db --default-artifact-root ./mlruns`,
   and `export MLFLOW_TRACKING_URI=http://localhost:5000`, since `config.yaml` names the Docker host.

3. **Airflow.** `export AIRFLOW_HOME=~/airflow AEROSCRAPE_HOME=$PWD`, link `dags/` into
   `$AIRFLOW_HOME/dags`, run `airflow db migrate`, create a user, then start `airflow webserver`
   and `airflow scheduler`.

4. **API and demo.**

   ```bash
   uvicorn aeroscrape.serving.api:app --port 8000
   BACKEND_URL=http://localhost:8000/v1/predict streamlit run src/aeroscrape/demo/streamlit_app.py
   ```

   Run Streamlit from the repository root, so it reads `.streamlit/config.toml` (theme and font).

## The `aeroscrape` command

One scrape, CSV import or export, outside Airflow:

```bash
aeroscrape                                   # scrape every airport and load into Postgres
aeroscrape --csv flights.csv --date 2026-06-03
aeroscrape --export-csv-path data/flights.csv
```

To parse saved pages again after a parser fix:

```python
from pathlib import Path
from aeroscrape.scraping.scraper import reparse

result = reparse(Path("data/raw_html/2026-11-07/090012"))
result.flights, result.failures
```

## Make targets

| Target | What |
| --- | --- |
| `setup`, `lint`, `format`, `test` | Development |
| `check-monitoring` | Prometheus config check and alert rule tests (Docker) |
| `data`, `bts-data` | Download the archive / the configured BTS months |
| `evaluate`, `learning-curve` | Local evaluation; `SOURCE=bts` for the US data |
| `up`, `down`, `up-offline`, `build-offline` | The Docker stack; the offline pair builds from the local uv cache when PyPI is unreachable |
| `migrate` | Add tables, roles and columns introduced after the database was created (safe to repeat) |
| `ingest`, `train`, `monitor` | Run a DAG now; `train` takes `TRAIN_FRACTION` and `SOURCE` |
| `model-status`, `rollback VERSION=n` | What serves; serve an earlier version |
| `replay DAYS=7 [NIGHT=1]` | Scraped departures through the API as live traffic; `NIGHT=1` moves them to night hours (a drift test) |
| `loadtest USERS=8` | Locust, three scenarios, one table |
| `fix-permissions` | Give back files that older root containers wrote into mounted folders |
