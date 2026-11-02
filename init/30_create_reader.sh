#!/bin/bash
# The API only reads flights for its feature lookup, so it gets a role that can do nothing else.
set -euo pipefail

psql -v ON_ERROR_STOP=1 -q --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" \
    -v reader="$READER_DB_USER" -v reader_password="$READER_DB_PASSWORD" -v app_db="$POSTGRES_DB" <<'SQL'
CREATE ROLE :"reader" LOGIN PASSWORD :'reader_password';
GRANT CONNECT ON DATABASE :"app_db" TO :"reader";
GRANT USAGE ON SCHEMA public TO :"reader";
GRANT SELECT ON airports, arrivals, departures TO :"reader";
SQL
