"""Airflow task callables for the monitor DAG: outcomes, live performance, drift, retraining."""

import logging
from datetime import UTC, datetime, timedelta
from typing import Any

import mlflow
import pandas as pd
from pandas.api.types import is_numeric_dtype

from aeroscrape.config import get_app_config, local_now, resolve_path
from aeroscrape.data.clean import clean_flights
from aeroscrape.models.registry import CHAMPION, alias_version, client_and_name
from aeroscrape.monitoring.drift import detect_drift, drift_html
from aeroscrape.monitoring.performance import (
    actual_delays,
    daily_performance,
    degraded_days,
    match_outcomes,
    offline_metrics,
    summary_for,
)
from aeroscrape.monitoring.push import GaugeValue, push_gauges
from aeroscrape.monitoring.reference import reference_inputs
from aeroscrape.monitoring.store import MonitoringStore
from aeroscrape.storage.database import Database

logger = logging.getLogger(__name__)

TRAIN_DAG = "train_delay_model"
MONITOR_DAG = "monitor_model"
DRIFT_EXPERIMENT = "flight_delay_monitoring"


def _settings() -> tuple[dict[str, Any], dict[str, Any]]:
    config = get_app_config()
    return config, config["monitoring"]


def task_join_outcomes(**kwargs):
    """Match predictions of flights that should have departed by now to their actual delays."""
    _, mon = _settings()
    db = Database()
    store = MonitoringStore(db)
    until = pd.Timestamp(local_now()) - pd.Timedelta(hours=mon["outcome_delay_hours"])
    since = until.normalize() - pd.Timedelta(days=mon["lookback_days"] + 1)
    pending = store.unmatched_predictions(since.to_pydatetime(), until.to_pydatetime())
    if pending.empty:
        logger.info("No unmatched predictions scheduled between %s and %s", since, until)
        return
    times = pd.to_datetime(pending["scheduled_time"])
    raw = db.fetch_flights(times.min().normalize() - pd.Timedelta(days=1), times.max() + pd.Timedelta(days=1))
    actual = actual_delays(clean_flights(raw)) if len(raw) else pd.DataFrame(columns=["flight_key", "delay_minutes"])
    inserted = store.insert_outcomes(match_outcomes(pending, actual))
    logger.info("Matched %d of %d pending predictions to an actual delay", inserted, len(pending))


def _offline(client, name: str, version: str) -> dict[str, float | None]:
    run = client.get_run(client.get_model_version(name, version).run_id)
    variant = run.data.tags.get("served_variant") or "booking"
    return offline_metrics(run.data.metrics, variant)


def task_live_performance(**kwargs):
    """Daily live MAE, Brier and coverage per model version, stored and pushed to Prometheus."""
    config, mon = _settings()
    store = MonitoringStore(Database())
    until = pd.Timestamp(local_now()) - pd.Timedelta(hours=mon["outcome_delay_hours"])
    since = until.normalize() - pd.Timedelta(days=mon["lookback_days"])
    rows = store.scored_flights(since.to_pydatetime(), until.to_pydatetime())
    daily = daily_performance(rows, mon["min_daily_outcomes"], config["pipeline"]["delay_threshold_minutes"])

    client, name = client_and_name(config)
    production = alias_version(client, name, CHAMPION)
    versions = set(daily["model_version"]) | ({production} if production else set())
    offline = {v: _offline(client, name, v) for v in versions}
    store.upsert_daily(daily, offline)

    streak, latest = 0, None
    if production:
        streak = degraded_days(
            daily[daily["model_version"] == production], offline[production]["mae"], mon["mae_tolerance"]
        )
        latest = summary_for(daily, production)
    logger.info("Live performance by day:\n%s", daily.to_string(index=False) if len(daily) else "(no scored flights)")

    labels = {"model_version": production or "none"}
    values = [
        GaugeValue("live_mae_degraded_days", "Consecutive recent days with live MAE above tolerance", streak, labels),
        GaugeValue(
            "monitor_performance_last_run_timestamp_seconds", "When live performance ran", datetime.now(UTC).timestamp()
        ),
    ]
    if latest is not None:
        off = offline[production]
        pairs = [
            ("live_mae_minutes", "Live MAE of the delay median, latest day", latest["mae"]),
            ("live_brier", "Live Brier score of the delay probability, latest day", latest["brier"]),
            ("live_interval_coverage", "Share of flights inside the 10-90% range, latest day", latest["coverage"]),
            ("live_matched_flights", "Flights with a known actual delay, latest day", latest["matched_flights"]),
            ("live_no_outcome_share", "Share of predicted flights without an actual delay", latest["no_outcome_share"]),
            (
                "live_no_inbound_share",
                "Share of predicted flights without an inbound match",
                latest["no_inbound_share"],
            ),
            ("offline_mae_minutes", "Test-set MAE of the served model", off["mae"]),
            ("offline_brier", "Test-set Brier score of the served model", off["brier"]),
            ("offline_interval_coverage", "Test-set coverage of the served model", off["coverage"]),
        ]
        values += [GaugeValue(n, h, float(v), labels) for n, h, v in pairs if v is not None and not pd.isna(v)]
    push_gauges("monitor_performance", values, config)
    kwargs["ti"].xcom_push(key="degraded_days", value=streak)


def task_detect_drift(**kwargs):
    """Compare recent model inputs with the production model's training inputs."""
    config, mon = _settings()
    drift_cfg = mon["drift"]
    since = datetime.now(UTC) - timedelta(days=mon["lookback_days"])
    current = MonitoringStore(Database()).recent_inputs(since)
    client, name = client_and_name(config)
    production = alias_version(client, name, CHAMPION)
    values = [GaugeValue("drift_current_rows", "Recent predictions compared", len(current))]
    if production is None or len(current) < drift_cfg["min_rows"]:
        logger.info("%d recent predictions; %d needed to check drift", len(current), drift_cfg["min_rows"])
        push_gauges("monitor_drift", values, config)
        kwargs["ti"].xcom_push(key="drifted", value=[])
        return

    reference = reference_inputs(config, client, name, production)
    numerical = {c for c in reference.columns if is_numeric_dtype(reference[c])}
    results = detect_drift(reference, current, drift_cfg["features"], numerical, drift_cfg["psi_threshold"])
    drifted = [r.feature for r in results if r.drifted]
    for r in results:
        logger.info("Drift %-28s PSI %.3f%s", r.feature, r.psi, "  DRIFT" if r.drifted else "")

    day = datetime.now(UTC).strftime("%Y-%m-%d")
    title = f"Input drift: model version {production}, last {mon['lookback_days']} days vs. training ({day})"
    report = drift_html(results, title, drift_cfg["psi_threshold"])
    path = resolve_path("reports/monitoring") / f"drift_{day}.html"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(DRIFT_EXPERIMENT)
    with mlflow.start_run(run_name="input_drift"):
        mlflow.set_tags({"model_version": production, "airflow_run_id": kwargs["run_id"], "drifted": ",".join(drifted)})
        mlflow.log_metrics({f"psi.{r.feature}": r.psi for r in results} | {"current_rows": len(current)})
        mlflow.log_text(report, "drift/report.html")
    logger.info("Drift report: %s (and in MLflow, experiment %s)", path, DRIFT_EXPERIMENT)

    labels = {"model_version": production}
    values += [
        GaugeValue(
            "feature_drift_psi", "PSI of a model input, recent vs. training", r.psi, labels | {"feature": r.feature}
        )
        for r in results
    ]
    values.append(
        GaugeValue(
            "drift_features_over_threshold", "Model inputs whose PSI is above the threshold", len(drifted), labels
        )
    )
    push_gauges("monitor_drift", values, config)
    kwargs["ti"].xcom_push(key="drifted", value=drifted)


def retrain_reasons(degraded: int, drifted: list[str], retrain_cfg: dict[str, Any]) -> list[str]:
    reasons = []
    if degraded >= retrain_cfg["degraded_days"]:
        reasons.append(f"live MAE degraded for {degraded} days")
    if drifted:
        reasons.append(f"input drift in {', '.join(drifted)}")
    return reasons


def _recently_triggered(cooldown_days: int) -> bool:
    from airflow.models import DagRun

    since = datetime.now(UTC) - timedelta(days=cooldown_days)
    return any(
        (run.conf or {}).get("triggered_by") == MONITOR_DAG and run.execution_date >= since
        for run in DagRun.find(dag_id=TRAIN_DAG)
    )


def decide_retrain(**kwargs) -> bool:
    """ShortCircuit: True triggers the training DAG, whose gate still decides about promotion."""
    _, mon = _settings()
    ti = kwargs["ti"]
    reasons = retrain_reasons(
        ti.xcom_pull(task_ids="live_performance", key="degraded_days") or 0,
        ti.xcom_pull(task_ids="detect_drift", key="drifted") or [],
        mon["retrain"],
    )
    if not reasons:
        logger.info("No alert; no retraining")
        return False
    if not mon["retrain"]["enabled"]:
        logger.warning("Alert (%s), but retraining on alerts is disabled", "; ".join(reasons))
        return False
    if _recently_triggered(mon["retrain"]["cooldown_days"]):
        logger.warning(
            "Alert (%s), but retraining was triggered in the last %d days",
            "; ".join(reasons),
            mon["retrain"]["cooldown_days"],
        )
        return False
    ti.xcom_push(key="reason", value="; ".join(reasons))
    logger.warning("Triggering retraining: %s", "; ".join(reasons))
    return True


def task_apply_retention(**kwargs):
    _, mon = _settings()
    cutoff = datetime.now(UTC) - timedelta(days=mon["retention_days"])
    deleted = MonitoringStore(Database()).delete_older_than(cutoff)
    logger.info("Deleted %d predictions older than %d days", deleted, mon["retention_days"])
