import pandas as pd
import pytest

from aeroscrape.data.split import Splits
from aeroscrape.models.lineage import collect_lineage, file_sha256, frame_sha256, git_state


def test_frame_hash_follows_content_not_index() -> None:
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    assert frame_sha256(df) == frame_sha256(df.set_axis([10, 11]))
    assert frame_sha256(df) != frame_sha256(df.assign(a=[1, 3]))


def test_git_state_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GIT_COMMIT", "abc123")
    monkeypatch.setenv("GIT_DIRTY", "false")
    assert git_state() == ("abc123", "false")


def test_collect_lineage(tmp_path) -> None:
    raw = tmp_path / "raw.csv"
    raw.write_text("a,b\n1,2\n")
    frame = pd.DataFrame(
        {"t": pd.date_range("2025-06-01", periods=6, freq="D"), "f": range(6), "y": [1.0, 2, 3, 4, 5, 6]}
    )
    splits = Splits(frame.iloc[:3], frame.iloc[3:4], frame.iloc[4:])

    lineage = collect_lineage(raw, splits, ["f"], "t", ["y"], git=("abc123", "false"))

    assert lineage["git_commit"] == "abc123"
    assert lineage["raw_data_sha256"] == file_sha256(raw)
    assert lineage["training_rows"] == "4" and lineage["test_rows"] == "2"
    assert lineage["training_to"] == "2025-06-04 00:00:00"
    assert lineage["features"] == "f"
