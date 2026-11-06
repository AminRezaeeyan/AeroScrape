"""How test error changes with the amount of training data.

Train on the last 1, 2, 4, 8 and 16 weeks before a fixed test window and score each on that
window. Plotted against the number of training departures, it answers "how much data would
the Iranian model need?" with evidence rather than a guess. Each point uses default
parameters (no tuning) and the 1-hour-before feature set, so only the data volume changes.

    python -m aeroscrape.models.learning_curve --source bts
"""

import argparse
import logging
from dataclasses import dataclass
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from aeroscrape.config import apply_profile, get_app_config, resolve_path  # noqa: E402
from aeroscrape.data.sources import load_cleaned_flights  # noqa: E402
from aeroscrape.data.split import recent_cut_dates  # noqa: E402
from aeroscrape.features.preprocess import make_model_frame, prepare_input, variant_feature_set  # noqa: E402
from aeroscrape.models.baselines import GroupMedianRegressor  # noqa: E402
from aeroscrape.models.evaluate import bootstrap_indices  # noqa: E402
from aeroscrape.models.gate import paired_mae_difference  # noqa: E402
from aeroscrape.models.plots import GRID, SERIES, SURFACE, TEXT_PRIMARY, style_axes  # noqa: E402
from aeroscrape.models.train import build_quantile_regressor  # noqa: E402

logger = logging.getLogger(__name__)


@dataclass
class CurvePoint:
    weeks: int
    train_rows: int
    model_mae: float
    baseline_mae: float
    change: float
    change_low: float
    change_high: float


def learning_curve(frame: pd.DataFrame, config: dict[str, Any], weeks: list[int], variant: str) -> list[CurvePoint]:
    eval_cfg, pipeline_cfg = config["evaluation"], config["pipeline"]
    time_col = eval_cfg["time_column"]
    target = pipeline_cfg["target_regression"]
    # The same test window as the main evaluation: the last `test_dates` dates with data
    test_start = recent_cut_dates(frame[time_col], eval_cfg["test_dates"], 1)[1]
    test = frame[frame[time_col] >= test_start]
    features = variant_feature_set(config, variant)
    X_test = prepare_input(test, features)
    y_test = test[target].to_numpy()
    indices = bootstrap_indices(len(test), eval_cfg["bootstrap_samples"], pipeline_cfg["random_state"])

    points = []
    for w in sorted(weeks):
        start = pd.Timestamp(test_start) - pd.Timedelta(weeks=w)
        train = frame[(frame[time_col] >= start) & (frame[time_col] < test_start)]
        if train[time_col].min() > start + pd.Timedelta(days=3):
            logger.info("Skipping %d weeks: the data doesn't reach back that far", w)
            continue
        model = build_quantile_regressor(pipeline_cfg, dict(pipeline_cfg["lgbm_params"]), features, 0.5)
        model.fit(prepare_input(train, features), train[target])
        prediction = model.predict(X_test)
        baseline = GroupMedianRegressor(["airline", "scheduled_time_of_day"]).fit(train, train[target]).predict(test)
        change, low, high = paired_mae_difference(y_test, prediction, baseline, indices, eval_cfg["confidence_level"])
        points.append(
            CurvePoint(
                w,
                len(train),
                float(np.mean(np.abs(y_test - prediction))),
                float(np.mean(np.abs(y_test - baseline))),
                change,
                low,
                high,
            )
        )
        logger.info(
            "%2d weeks, %d rows: MAE %.2f (baseline %.2f)", w, len(train), points[-1].model_mae, points[-1].baseline_mae
        )
    return points


def curve_figure(points: list[CurvePoint], title: str, reference_rows: int | None = None) -> Figure:
    """Test MAE against training departures (log scale), for the model and the baseline."""
    fig, ax = plt.subplots(figsize=(7, 4.6), dpi=150, facecolor=SURFACE)
    style_axes(ax, title, "Training departures (log scale)", "Test MAE (minutes)")
    rows = [p.train_rows for p in points]
    for (label, values), color in zip(
        [
            ("LightGBM", [p.model_mae for p in points]),
            ("Airline × time-of-day median", [p.baseline_mae for p in points]),
        ],
        SERIES,
        strict=False,
    ):
        ax.plot(
            rows,
            values,
            color=color,
            linewidth=2,
            marker="o",
            markersize=6,
            label=label,
            markeredgecolor=SURFACE,
            markeredgewidth=1.5,
            zorder=3,
        )
        ax.annotate(
            label,
            (rows[-1], values[-1]),
            xytext=(8, 0),
            textcoords="offset points",
            color=TEXT_PRIMARY,
            fontsize=8,
            va="center",
        )
    for p in points:
        ax.annotate(
            f"{p.weeks} wk",
            (p.train_rows, p.model_mae),
            xytext=(0, -14),
            textcoords="offset points",
            ha="center",
            fontsize=7,
            color=TEXT_PRIMARY,
        )
    values = [p.model_mae for p in points] + [p.baseline_mae for p in points]
    pad = (max(values) - min(values)) * 0.15 or 1.0
    ax.set_ylim(min(values) - pad, max(values) + pad)
    if reference_rows:
        ax.axvline(reference_rows, color=GRID, linewidth=1.5, linestyle=(0, (4, 3)), zorder=1)
        ax.text(
            reference_rows,
            0.5,
            " Iranian training rows",
            transform=ax.get_xaxis_transform(),
            rotation=90,
            va="center",
            ha="right",
            fontsize=7,
            color=TEXT_PRIMARY,
        )
    ax.set_xscale("log")
    # Include the reference line even when it lies below the smallest training set
    ax.set_xlim(min([*rows, reference_rows or rows[0]]) / 1.5, max(rows) * 3)
    # The space between the two lines is empty, so the legend goes there
    ax.legend(loc="center right", frameon=False, fontsize=9, labelcolor=TEXT_PRIMARY)
    fig.tight_layout()
    return fig


def curve_markdown(points: list[CurvePoint], source: str) -> str:
    lines = [
        f"# Learning curve ({source})",
        "",
        "Each row trains on the given number of weeks immediately before the same test window, with default "
        "parameters and the 1-hour-before features. The change is LightGBM minus the airline × time-of-day "
        "baseline trained on the same weeks, with a paired 95% bootstrap interval. Chart: `learning_curve.png`.",
        "",
        "| Weeks | Training departures | LightGBM MAE | Baseline MAE | Change vs baseline |",
        "| --- | --- | --- | --- | --- |",
    ]
    for p in points:
        lines.append(
            f"| {p.weeks} | {p.train_rows:,} | {p.model_mae:.2f} | {p.baseline_mae:.2f} | "
            f"{p.change:+.2f} [{p.change_low:+.2f}, {p.change_high:+.2f}] |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Test MAE against the amount of training data.")
    parser.add_argument("--source", default="bts", help="Data source profile from config.yaml")
    parser.add_argument("--weeks", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--variant", default="pre_departure")
    parser.add_argument("--iran-rows", type=int, default=2887, help="Iranian training rows, drawn as a reference")
    args = parser.parse_args()

    config = apply_profile(get_app_config(), args.source)
    flights, _ = load_cleaned_flights(config)
    points = learning_curve(make_model_frame(flights, config), config, args.weeks, args.variant)
    out_dir = resolve_path(config["evaluation"]["report_path"]).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "learning_curve.md").write_text(curve_markdown(points, args.source), encoding="utf-8")
    fig = curve_figure(points, f"Test error by amount of training data ({args.source.upper()})", args.iran_rows)
    fig.savefig(out_dir / "learning_curve.png", facecolor=fig.get_facecolor())
    print(curve_markdown(points, args.source))


if __name__ == "__main__":
    main()
