"""What a model was built from, recorded as MLflow tags and in the model's metadata.

With these a model can be traced back to its code and data: the commit, whether the
working tree had uncommitted changes, a hash of the raw file, a hash of the exact
training frame, the time range and row counts, and the feature list.
"""

import hashlib
import logging
import os
import subprocess
from pathlib import Path

import pandas as pd

from aeroscrape.config import project_root
from aeroscrape.data.split import Splits

logger = logging.getLogger(__name__)


def git_state() -> tuple[str, str]:
    """(commit, dirty) from GIT_COMMIT/GIT_DIRTY, else from git; "unknown" when neither is available.

    Containers have no .git directory, so `make train` passes the commit in the DAG run's conf.
    """
    if os.getenv("GIT_COMMIT"):
        return os.environ["GIT_COMMIT"], os.getenv("GIT_DIRTY", "unknown")
    try:
        root = str(project_root())
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return commit, "true" if status else "false"
    except (OSError, subprocess.CalledProcessError):
        return "unknown", "unknown"


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frame_sha256(df: pd.DataFrame) -> str:
    """A content hash of a DataFrame's values, independent of its index."""
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=False).to_numpy().tobytes()).hexdigest()


def collect_lineage(
    raw_path: str | Path | None,
    splits: Splits,
    features: list[str],
    time_column: str,
    targets: list[str],
    git: tuple[str, str] | None = None,
) -> dict[str, str]:
    commit, dirty = git or git_state()
    fit = splits.train_val
    lineage = {
        "git_commit": commit,
        "git_dirty": dirty,
        "raw_data_sha256": file_sha256(raw_path) if raw_path and Path(raw_path).exists() else "unknown",
        "training_data_sha256": frame_sha256(fit[[*features, *targets]]),
        "training_rows": str(len(fit)),
        "test_rows": str(len(splits.test)),
        "training_from": str(fit[time_column].min()),
        "training_to": str(fit[time_column].max()),
        "test_from": str(splits.test[time_column].min()),
        "test_to": str(splits.test[time_column].max()),
        "features": ",".join(features),
    }
    if dirty == "true":
        logger.warning("Training from a working tree with uncommitted changes (commit %s)", commit)
    return lineage


def lineage_markdown(lineage: dict[str, str]) -> list[str]:
    rows = [f"| {key} | `{value}` |" for key, value in lineage.items() if key != "features"]
    return ["## Lineage", "", "| Field | Value |", "| --- | --- |", *rows, ""]
