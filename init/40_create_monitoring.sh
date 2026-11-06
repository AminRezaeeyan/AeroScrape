#!/bin/bash
# Monitoring tables: the API's prediction log, the outcomes matched to it later, and the daily
# live metrics. Safe to run again: on an existing database, `make migrate` runs it.
#
#   API (reader role)       inserts predictions; reads nothing it didn't already
#   app role (Airflow)      reads predictions, writes outcomes and live metrics, deletes old rows
#   Grafana role            reads the live metrics only
set -euo pipefail

psql -v ON_ERROR_STOP=1 -q --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" \
    -v app_user="$APP_DB_USER" -v reader="$READER_DB_USER" \
    -v grafana_user="$GRAFANA_DB_USER" -v grafana_password="$GRAFANA_DB_PASSWORD" -v app_db="$POSTGRES_DB" <<'SQL'
CREATE TABLE IF NOT EXISTS predictions (
    id BIGSERIAL PRIMARY KEY,
    request_id TEXT NOT NULL,
    traffic_source TEXT NOT NULL DEFAULT 'client',   -- client, replay or loadtest
    flight_key TEXT,                                 -- airport|flight number|schedule; null without a flight number
    airport TEXT NOT NULL,
    airline TEXT NOT NULL,
    flight_number TEXT,
    scheduled_time TIMESTAMP NOT NULL,
    model_version TEXT NOT NULL,
    p_delay REAL NOT NULL,
    likely_delayed BOOLEAN NOT NULL,
    q10 REAL NOT NULL,
    q50 REAL NOT NULL,
    q90 REAL NOT NULL,
    features JSONB NOT NULL,                         -- the model input, for drift checks
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_predictions_created_at ON predictions (created_at);
CREATE INDEX IF NOT EXISTS idx_predictions_scheduled_time ON predictions (scheduled_time);
CREATE INDEX IF NOT EXISTS idx_predictions_flight_key ON predictions (flight_key);

CREATE TABLE IF NOT EXISTS prediction_outcomes (
    prediction_id BIGINT PRIMARY KEY REFERENCES predictions (id) ON DELETE CASCADE,
    actual_delay_min REAL NOT NULL,
    matched_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS live_metrics_daily (
    day DATE NOT NULL,
    model_version TEXT NOT NULL,
    predicted_flights INT NOT NULL,      -- distinct flights with a flight number
    matched_flights INT NOT NULL,        -- of those, with a known actual delay
    mae REAL,
    brier REAL,
    coverage REAL,
    no_outcome_share REAL,
    no_inbound_share REAL,
    offline_mae REAL,
    offline_brier REAL,
    offline_coverage REAL,
    computed_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (day, model_version)
);

GRANT INSERT ON predictions TO :"reader";
GRANT USAGE ON SEQUENCE predictions_id_seq TO :"reader";
GRANT SELECT, DELETE ON predictions TO :"app_user";
GRANT SELECT, INSERT, UPDATE ON prediction_outcomes, live_metrics_daily TO :"app_user";

SELECT format('CREATE ROLE %I LOGIN PASSWORD %L', :'grafana_user', :'grafana_password')
WHERE NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = :'grafana_user') \gexec
GRANT CONNECT ON DATABASE :"app_db" TO :"grafana_user";
GRANT USAGE ON SCHEMA public TO :"grafana_user";
GRANT SELECT ON live_metrics_daily TO :"grafana_user";
SQL
