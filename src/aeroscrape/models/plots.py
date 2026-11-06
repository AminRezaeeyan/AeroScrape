"""Static charts for the evaluation report (PNG, light background, for README and MLflow).

Colors are the first three categorical slots of the validated reference palette;
they pass the colorblind checks as a set. Aqua is below 3:1 contrast on the light
surface, so every series is also labeled directly and the report repeats the data
as a table.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e6e5e1"
REFERENCE = "#8a8984"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")  # blue, orange, aqua: fixed order, never cycled


def style_axes(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=12, color=TEXT_PRIMARY, pad=12)
    ax.set_xlabel(xlabel, color=TEXT_SECONDARY, fontsize=10)
    ax.set_ylabel(ylabel, color=TEXT_SECONDARY, fontsize=10)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9, length=0)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)


def calibration_bins(y: np.ndarray, proba: np.ndarray, bins: int = 10) -> list[tuple[float, float, int]]:
    """(mean predicted, observed rate, count) for each non-empty equal-width probability bin."""
    which = np.minimum((proba * bins).astype(int), bins - 1)
    return [
        (float(proba[which == b].mean()), float(y[which == b].mean()), int((which == b).sum()))
        for b in range(bins)
        if (which == b).any()
    ]


def reliability_figure(y: np.ndarray, curves: dict[str, np.ndarray], title: str) -> Figure:
    """Predicted probability against observed delay rate; the diagonal is perfect calibration."""
    if len(curves) > len(SERIES):
        raise ValueError(f"At most {len(SERIES)} series; fold the rest into a table")
    fig, ax = plt.subplots(figsize=(7, 5.2), dpi=150, facecolor=SURFACE)
    style_axes(ax, title, "Predicted probability of a delay over 15 min", "Observed share delayed")
    ax.plot([0, 1], [0, 1], color=REFERENCE, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
    ax.text(0.97, 0.93, "perfect calibration", color=TEXT_SECONDARY, fontsize=8, ha="right", rotation=36)

    ends = []
    for (name, proba), color in zip(curves.items(), SERIES, strict=False):
        points = calibration_bins(y, proba)
        xs, ys = [p[0] for p in points], [p[1] for p in points]
        ax.plot(
            xs,
            ys,
            color=color,
            linewidth=2,
            marker="o",
            markersize=6,
            label=name,
            zorder=3,
            markeredgecolor=SURFACE,  # surface ring where markers overlap
            markeredgewidth=1.5,
        )
        ends.append((ys[-1], xs[-1], name))

    # Direct labels in the right margin, spread apart so lines that end together stay readable
    min_gap = 0.06
    placed: list[float] = []
    for end_y, end_x, name in sorted(ends):
        label_y = max(end_y, placed[-1] + min_gap) if placed else end_y
        placed.append(label_y)
        ax.annotate(
            name,
            xy=(end_x, end_y),
            xytext=(1.03, label_y),
            textcoords="data",
            color=TEXT_PRIMARY,
            fontsize=8,
            va="center",
            arrowprops={"arrowstyle": "-", "color": GRID, "linewidth": 0.8},
        )

    ax.set_xlim(0, 1.32)
    ax.set_ylim(0, 1.02)
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.legend(loc="upper left", frameon=False, fontsize=9, labelcolor=TEXT_PRIMARY)
    fig.tight_layout()
    return fig


def importance_figure(names: list[str], values: list[float], title: str, xlabel: str) -> Figure:
    """Horizontal bars, largest at the top; one series, so no legend."""
    order = np.argsort(values)
    fig, ax = plt.subplots(figsize=(7, 0.42 * len(names) + 1.4), dpi=150, facecolor=SURFACE)
    style_axes(ax, title, xlabel, "")
    ax.grid(axis="y", visible=False)
    ax.barh([names[i] for i in order], [values[i] for i in order], color=SERIES[0], height=0.62)
    for rank, i in enumerate(order):
        ax.text(values[i], rank, f" {values[i]:.2f}", va="center", fontsize=8, color=TEXT_SECONDARY)
    ax.set_xlim(0, max(values) * 1.15)
    fig.tight_layout()
    return fig
