#!/bin/bash
# Airflow metadata gets its own database so its tables stay apart from the flight data.
# Postgres only runs init scripts on an empty data volume.
set -euo pipefail

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" \
    -c "CREATE DATABASE \"${AIRFLOW_DB_NAME}\""
