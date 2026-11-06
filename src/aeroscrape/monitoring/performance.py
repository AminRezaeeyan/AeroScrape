"""Live performance: logged predictions against the delays that actually happened.

The actual delay comes from the same cleaner and the same row selection as the training
labels (departed domestic departures with a plausible delay), so live and offline numbers
measure the same thing. Each flight counts once per model version: its latest prediction.
"""

from typing import Any

import numpy as np
import pandas as pd

from aeroscrape.data.clean import select_training_rows
from aeroscrape.models.evaluate import _slug
from aeroscrape.models.experiment import classifier_name, interval_name, regressor_name
from aeroscrape.monitoring.prediction_log import flight_key

METRICS = ("mae", "brier", "coverage")


def actual_delays(cleaned: pd.DataFrame) -> pd.DataFrame:
    """flight_key -> delay_minutes for flights that departed, labelled exactly as for training."""
    departed = select_training_rows(cleaned)
    keys = [
        flight_key(a, n, s)
        for a, n, s in zip(departed["airport"], departed["flight_number"], departed["scheduled_datetime"], strict=True)
    ]
    return pd.DataFrame({"flight_key": keys, "delay_minutes": departed["delay_minutes"].to_numpy()}).drop_duplicates(
        "flight_key"
    )


def match_outcomes(predictions: pd.DataFrame, actual: pd.DataFrame) -> pd.DataFrame:
    """(prediction_id, actual_delay_min) for predictions whose flight has a known delay."""
    matched = predictions[["id", "flight_key"]].merge(actual, on="flight_key")
    return matched.rename(columns={"id": "prediction_id", "delay_minutes": "actual_delay_min"})[
        ["prediction_id", "actual_delay_min"]
    ]


def daily_performance(rows: pd.DataFrame, min_outcomes: int, delay_threshold: float = 15) -> pd.DataFrame:
    """Per scheduled day and model version: flights, matched flights, MAE, Brier, coverage, shares.

    `rows` has one row per logged prediction: flight_key, model_version, scheduled_time,
    created_at, p_delay, q10, q50, q90, has_inbound and actual_delay_min (NaN if unknown).
    A day with fewer than `min_outcomes` matched flights gets no MAE, Brier or coverage.
    """
    columns = ["day", "model_version", "predicted_flights", "matched_flights", *METRICS]
    columns += ["no_outcome_share", "no_inbound_share"]
    if rows.empty:
        return pd.DataFrame(columns=columns)
    latest = rows.sort_values("created_at").drop_duplicates(["model_version", "flight_key"], keep="last")
    latest = latest.assign(day=pd.to_datetime(latest["scheduled_time"]).dt.date)

    out = []
    for (day, version), g in latest.groupby(["day", "model_version"]):
        known = g[g["actual_delay_min"].notna()]
        actual = known["actual_delay_min"].to_numpy(float)
        enough = len(known) >= min_outcomes
        delayed = (actual > delay_threshold).astype(float)
        inside = (known["q10"].to_numpy() <= actual) & (actual <= known["q90"].to_numpy())
        out.append(
            {
                "day": day,
                "model_version": str(version),
                "predicted_flights": len(g),
                "matched_flights": len(known),
                "mae": float(np.mean(np.abs(known["q50"].to_numpy() - actual))) if enough else None,
                "brier": float(np.mean((known["p_delay"].to_numpy() - delayed) ** 2)) if enough else None,
                "coverage": float(np.mean(inside)) if enough else None,
                "no_outcome_share": 1 - len(known) / len(g),
                "no_inbound_share": float((pd.to_numeric(g["has_inbound"], errors="coerce").fillna(0) == 0).mean()),
            }
        )
    return pd.DataFrame(out, columns=columns).sort_values(["model_version", "day"]).reset_index(drop=True)


def degraded_days(daily: pd.DataFrame, offline_mae: float | None, tolerance: float) -> int:
    """Consecutive most recent days (with enough outcomes) whose MAE exceeds offline MAE by `tolerance`."""
    if offline_mae is None or daily.empty:
        return 0
    streak = 0
    for mae in daily.sort_values("day", ascending=False)["mae"]:
        if mae is None or pd.isna(mae):
            continue  # too few outcomes to judge this day
        if mae > offline_mae * (1 + tolerance):
            streak += 1
        else:
            break
    return streak


def offline_metrics(run_metrics: dict[str, float], variant: str) -> dict[str, float | None]:
    """The served variant's test-set MAE, Brier and coverage, from its training run's metrics."""
    keys = {
        "mae": f"test/{_slug(regressor_name(variant))}/mae_min",
        "brier": f"test/{_slug(classifier_name(variant))}/brier",
        "coverage": f"test/{_slug(interval_name(variant))}/coverage_of_10_90_interval",
    }
    return {name: run_metrics.get(key) for name, key in keys.items()}


def summary_for(daily: pd.DataFrame, version: str) -> dict[str, Any] | None:
    """The most recent day with live metrics for a model version, or None."""
    rows = daily[(daily["model_version"] == version) & daily["mae"].notna()]
    if rows.empty:
        return None
    return {str(k): v for k, v in rows.sort_values("day").iloc[-1].items()}
