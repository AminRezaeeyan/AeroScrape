"""Shared setup for the analysis notebooks: chart style, the Persian font and data loading.

The notebooks use the project's own code (aeroscrape) for cleaning, features and models, so what
they show is what the pipeline does. Run them with `make notebooks` or open them in Jupyter.
"""

import warnings
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager

ASSETS = Path(__file__).parent / "assets"
BLUE, ORANGE, GREEN, RED, GREY = "#2563eb", "#f59e0b", "#16a34a", "#dc2626", "#94a3b8"


def setup() -> None:
    """Vazirmatn for every chart and a quiet, consistent style.

    Vazirmatn has Latin and Persian glyphs, and matplotlib 3.11 shapes and orders right-to-left
    text itself, so Persian airline and airport names are used as they are.
    """
    for font in ASSETS.glob("*.ttf"):
        font_manager.fontManager.addfont(str(font))
    plt.rcParams.update(
        {
            "font.family": "Vazirmatn",
            "figure.dpi": 110,
            "figure.figsize": (9, 4.2),
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.titleweight": "bold",
            "axes.titlesize": 12,
            "axes.titlelocation": "left",
            "axes.prop_cycle": plt.cycler(color=[BLUE, ORANGE, GREEN, RED, GREY]),
        }
    )
    pd.set_option("display.precision", 3)
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.max_colwidth", 80)
    # Library notices that say nothing about the analysis
    warnings.filterwarnings("ignore", message=".*TreeExplainer shap values output has changed.*")
    warnings.filterwarnings("ignore", message="IProgress not found")


def show_inline() -> None:
    """Back to inline figures: importing the training code selects the non-interactive Agg backend."""
    from IPython import get_ipython

    get_ipython().run_line_magic("matplotlib", "inline")
    setup()


def project_config() -> dict[str, Any]:
    """The project configuration, reading the 2025 archive only (no database needed)."""
    from aeroscrape.config import get_app_config

    config = get_app_config()
    config["data_sources"] = {**config["data_sources"], "database": False}
    return config


def load_flights() -> tuple[pd.DataFrame, Any]:
    """Every cleaned flight from the archive, and the cleaning audit."""
    from aeroscrape.data.clean import CleaningAudit
    from aeroscrape.data.sources import load_cleaned_flights

    audit = CleaningAudit()
    flights, _ = load_cleaned_flights(project_config(), audit)
    return flights, audit
