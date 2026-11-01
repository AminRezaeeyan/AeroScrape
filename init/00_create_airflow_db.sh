#!/bin/bash
# Airflow gets its own role and database, so it never touches the flight data.
# Postgres only runs init scripts on an empty data volume.
set -euo pipefail

psql -v ON_ERROR_STOP=1 -q --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" \
    -v af_user="$AIRFLOW_DB_USER" -v af_password="$AIRFLOW_DB_PASSWORD" -v af_db="$AIRFLOW_DB_NAME" <<'SQL'
CREATE ROLE :"af_user" LOGIN PASSWORD :'af_password';
CREATE DATABASE :"af_db" OWNER :"af_user";
REVOKE CONNECT ON DATABASE :"af_db" FROM PUBLIC;
\connect :"af_db"
GRANT ALL ON SCHEMA public TO :"af_user";
SQL
