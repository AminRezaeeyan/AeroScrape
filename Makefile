DATA_URL := https://github.com/AminRezaeeyan/AeroScrape/releases/download/data-v1/data.csv
DATA_SHA256 := 7b9ba6559b9813b8f1c306f4aab33d49c137e7d329d2d1fd3f54846ce7b5919f
DATA_FILE := data/raw/data.csv
TRAIN_DAG := train_delay_model
INGEST_DAG := ingest_flights
TRAIN_FRACTION ?= 1.0
SOURCE ?= iran_fids
USERS ?= 8
LOADTEST_SECONDS ?= 30
API_URL ?= http://localhost:8000

.PHONY: setup lint format test loadtest fix-permissions build-offline up-offline data bts-data evaluate learning-curve train ingest rollback model-status up down

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

evaluate:  ## Tune, train and evaluate locally on a time split; SOURCE=bts for the US data
	uv run --frozen python -m aeroscrape.models.experiment --source $(SOURCE)

learning-curve:  ## Test error against weeks of training data; SOURCE=bts by default
	uv run --frozen python -m aeroscrape.models.learning_curve --source $(if $(filter iran_fids,$(SOURCE)),bts,$(SOURCE))

bts-data:  ## Download the BTS months configured in config.yaml (needs access to transtats.bts.gov)
	uv run --frozen python -m aeroscrape.scraping.bts

train:  ## Train a challenger and run the promotion gate; TRAIN_FRACTION=0.1 trains a deliberately worse model
	docker compose exec airflow-scheduler airflow dags unpause $(TRAIN_DAG)
	docker compose exec airflow-scheduler airflow dags trigger $(TRAIN_DAG) \
		--conf '{"git_commit": "$(shell git rev-parse HEAD)", "git_dirty": "$(if $(shell git status --porcelain --untracked-files=no),true,false)", "train_fraction": $(TRAIN_FRACTION), "source": "$(SOURCE)"}'

ingest:  ## Scrape the flight boards now (the ingest DAG also runs every 3 hours once unpaused)
	docker compose exec airflow-scheduler airflow dags unpause $(INGEST_DAG)
	docker compose exec airflow-scheduler airflow dags trigger $(INGEST_DAG)

rollback:  ## Serve an earlier model version: make rollback VERSION=7
	@test -n "$(VERSION)" || (echo "usage: make rollback VERSION=<model version>" && exit 1)
	docker compose exec airflow-scheduler python -m aeroscrape.models.registry rollback $(VERSION)

model-status:  ## Show which model versions are production and challenger
	docker compose exec airflow-scheduler python -m aeroscrape.models.registry status

loadtest:  ## Locust against the running API (make up first): USERS=8, 30 s per scenario
	@mkdir -p reports/locust
	@export AEROSCRAPE_API_KEY="$$(grep -E '^AEROSCRAPE_API_KEY=' .env | cut -d= -f2-)"; \
	for scenario in SingleFlight ExplainedFlight BatchOfFlights; do \
		uv run --frozen locust -f tools/locustfile.py $$scenario --headless --only-summary \
			-u $(USERS) -r $(USERS) -t $(LOADTEST_SECONDS)s --host $(API_URL) \
			--csv reports/locust/$$scenario --loglevel WARNING || exit 1; \
	done
	@uv run --frozen python tools/locust_report.py $(USERS) \
		reports/locust/SingleFlight reports/locust/ExplainedFlight reports/locust/BatchOfFlights

fix-permissions:  ## Give back to you files that older containers wrote as root into mounted folders
	docker run --rm -v "$(CURDIR):/work" alpine chown -R $$(id -u):$$(id -g) /work/mlruns /work/logs /work/data /work/models

up:  ## Build and start the Docker stack
	docker compose up -d --build

build-offline:  ## Build the images from the local uv cache when PyPI is unreachable; then `make up-offline`
	tools/build_offline.sh

up-offline:  ## Start the stack from images already built (e.g. by build-offline)
	docker compose up -d --no-build

down:  ## Stop the Docker stack
	docker compose down
