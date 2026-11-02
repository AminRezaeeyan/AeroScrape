DATA_URL := https://github.com/AminRezaeeyan/AeroScrape/releases/download/data-v1/data.csv
DATA_SHA256 := 7b9ba6559b9813b8f1c306f4aab33d49c137e7d329d2d1fd3f54846ce7b5919f
DATA_FILE := data/raw/data.csv
TRAIN_DAG := train_delay_model
INGEST_DAG := ingest_flights
TRAIN_FRACTION ?= 1.0

.PHONY: setup lint format test data evaluate train ingest rollback model-status up down

setup:  ## Install all dependencies from the lock file and the git hooks
	uv sync --frozen --all-extras
	uv run --frozen pre-commit install

lint:  ## Check lint, formatting and types
	uv run --frozen ruff check .
	uv run --frozen ruff format --check .
	uv run --frozen mypy

format:  ## Fix lint issues and format the code
	uv run --frozen ruff check --fix .
	uv run --frozen ruff format .

test:  ## Run the test suite
	uv run --frozen pytest -q

data:  ## Download the raw dataset and verify its checksum
	mkdir -p $(dir $(DATA_FILE))
	curl -fsSL -o $(DATA_FILE) $(DATA_URL)
	echo "$(DATA_SHA256)  $(DATA_FILE)" | sha256sum --check

evaluate:  ## Tune, train and evaluate locally on a time split; writes reports/evaluation.md
	uv run --frozen python -m aeroscrape.models.experiment

train:  ## Train a challenger and run the promotion gate; TRAIN_FRACTION=0.1 trains a deliberately worse model
	docker compose exec airflow-scheduler airflow dags unpause $(TRAIN_DAG)
	docker compose exec airflow-scheduler airflow dags trigger $(TRAIN_DAG) \
		--conf '{"git_commit": "$(shell git rev-parse HEAD)", "git_dirty": "$(if $(shell git status --porcelain --untracked-files=no),true,false)", "train_fraction": $(TRAIN_FRACTION)}'

ingest:  ## Scrape the flight boards now (the ingest DAG also runs every 3 hours once unpaused)
	docker compose exec airflow-scheduler airflow dags unpause $(INGEST_DAG)
	docker compose exec airflow-scheduler airflow dags trigger $(INGEST_DAG)

rollback:  ## Serve an earlier model version: make rollback VERSION=7
	@test -n "$(VERSION)" || (echo "usage: make rollback VERSION=<model version>" && exit 1)
	docker compose exec airflow-scheduler python -m aeroscrape.models.registry rollback $(VERSION)

model-status:  ## Show which model versions are production and challenger
	docker compose exec airflow-scheduler python -m aeroscrape.models.registry status

up:  ## Build and start the Docker stack
	docker compose up -d --build

down:  ## Stop the Docker stack
	docker compose down
