"""SQL for the monitoring tables (created by init/40_create_monitoring.sh)."""

from datetime import datetime
from typing import Any

import pandas as pd
from psycopg2.extras import execute_values

# Load-test traffic exercises the service but says nothing about real inputs or outcomes
EXCLUDED_SOURCES = ("loadtest",)


class MonitoringStore:
    def __init__(self, source: Any):
        self.source = source  # anything with get_connection(), e.g. storage.database.Database

    def _query(self, sql: str, params: tuple) -> pd.DataFrame:
        conn = self.source.get_connection()
        try:
            with conn, conn.cursor() as cur:
                cur.execute(sql, params)
                return pd.DataFrame(cur.fetchall(), columns=[c.name for c in cur.description])
        finally:
            conn.close()

    def _execute(self, sql: str, rows: list[tuple] | None = None, params: tuple = ()) -> int:
        conn = self.source.get_connection()
        try:
            with conn, conn.cursor() as cur:
                if rows is not None:
                    # One page, so rowcount covers every row (execute_values reports the last page only)
                    execute_values(cur, sql, rows, page_size=max(len(rows), 1))
                else:
                    cur.execute(sql, params)
                return cur.rowcount
        finally:
            conn.close()

    def unmatched_predictions(self, since: datetime, until: datetime) -> pd.DataFrame:
        """Predictions with a flight number, scheduled in [since, until), not matched to an outcome yet."""
        return self._query(
            """
            SELECT p.id, p.flight_key, p.scheduled_time
            FROM predictions p LEFT JOIN prediction_outcomes o ON o.prediction_id = p.id
            WHERE o.prediction_id IS NULL AND p.flight_key IS NOT NULL
              AND p.scheduled_time >= %s AND p.scheduled_time < %s AND p.traffic_source <> ALL(%s)
            """,
            (since, until, list(EXCLUDED_SOURCES)),
        )

    def insert_outcomes(self, outcomes: pd.DataFrame) -> int:
        if outcomes.empty:
            return 0
        rows = [(int(i), float(d)) for i, d in outcomes[["prediction_id", "actual_delay_min"]].itertuples(index=False)]
        return self._execute(
            "INSERT INTO prediction_outcomes (prediction_id, actual_delay_min) VALUES %s "
            "ON CONFLICT (prediction_id) DO NOTHING",
            rows,
        )

    def scored_flights(self, since: datetime, until: datetime) -> pd.DataFrame:
        """Predictions with a flight number scheduled in [since, until), with the actual delay if known."""
        return self._query(
            """
            SELECT p.flight_key, p.model_version, p.scheduled_time, p.created_at, p.p_delay,
                   p.q10, p.q50, p.q90, p.features ->> 'has_inbound' AS has_inbound, o.actual_delay_min
            FROM predictions p LEFT JOIN prediction_outcomes o ON o.prediction_id = p.id
            WHERE p.flight_key IS NOT NULL AND p.scheduled_time >= %s AND p.scheduled_time < %s
              AND p.traffic_source <> ALL(%s)
            """,
            (since, until, list(EXCLUDED_SOURCES)),
        )

    def upsert_daily(self, daily: pd.DataFrame, offline: dict[str, dict[str, float | None]]) -> int:
        """Write daily live metrics; `offline` maps model version to its offline mae/brier/coverage."""
        if daily.empty:
            return 0

        def value(v: Any) -> Any:
            return None if v is None or pd.isna(v) else float(v)

        rows = []
        for r in daily.to_dict("records"):
            off = offline.get(r["model_version"], {})
            rows.append(
                (
                    r["day"],
                    r["model_version"],
                    int(r["predicted_flights"]),
                    int(r["matched_flights"]),
                    *(value(r[m]) for m in ("mae", "brier", "coverage", "no_outcome_share", "no_inbound_share")),
                    *(value(off.get(m)) for m in ("mae", "brier", "coverage")),
                )
            )
        return self._execute(
            """
            INSERT INTO live_metrics_daily (day, model_version, predicted_flights, matched_flights, mae, brier,
                coverage, no_outcome_share, no_inbound_share, offline_mae, offline_brier, offline_coverage)
            VALUES %s
            ON CONFLICT (day, model_version) DO UPDATE SET
                predicted_flights = EXCLUDED.predicted_flights, matched_flights = EXCLUDED.matched_flights,
                mae = EXCLUDED.mae, brier = EXCLUDED.brier, coverage = EXCLUDED.coverage,
                no_outcome_share = EXCLUDED.no_outcome_share, no_inbound_share = EXCLUDED.no_inbound_share,
                offline_mae = EXCLUDED.offline_mae, offline_brier = EXCLUDED.offline_brier,
                offline_coverage = EXCLUDED.offline_coverage, computed_at = now()
            """,
            rows,
        )

    def recent_inputs(self, since: datetime) -> pd.DataFrame:
        """The model inputs of predictions made since `since`, one column per feature."""
        rows = self._query(
            "SELECT features FROM predictions WHERE created_at >= %s AND traffic_source <> ALL(%s)",
            (since, list(EXCLUDED_SOURCES)),
        )
        return pd.DataFrame(rows["features"].tolist()) if len(rows) else pd.DataFrame()

    def delete_older_than(self, cutoff: datetime) -> int:
        return self._execute("DELETE FROM predictions WHERE created_at < %s", params=(cutoff,))
