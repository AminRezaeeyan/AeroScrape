from pathlib import Path

import pytest

from aeroscrape import config


@pytest.fixture(autouse=True)
def clear_config_cache():
    config.project_root.cache_clear()
    config.get_app_config.cache_clear()
    yield
    config.project_root.cache_clear()
    config.get_app_config.cache_clear()


def test_aeroscrape_home_overrides_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AEROSCRAPE_HOME", str(tmp_path))
    assert config.project_root() == tmp_path.resolve()
    assert config.resolve_path("models/") == tmp_path.resolve() / "models"


def test_root_is_found_from_a_subdirectory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AEROSCRAPE_HOME", raising=False)
    (tmp_path / "config.yaml").write_text("mlflow: {tracking_uri: x}\n")
    nested = tmp_path / "a" / "b"
    nested.mkdir(parents=True)
    monkeypatch.chdir(nested)
    assert config.project_root() == tmp_path.resolve()


def test_env_overrides_mlflow_uri(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AEROSCRAPE_HOME", str(tmp_path))
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://from-env:5000")
    (tmp_path / "config.yaml").write_text("mlflow: {tracking_uri: http://from-yaml:5000}\n")
    assert config.get_app_config()["mlflow"]["tracking_uri"] == "http://from-env:5000"


def test_loading_config_does_not_need_database_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AEROSCRAPE_HOME", str(tmp_path))
    for name in config.DB_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    (tmp_path / "config.yaml").write_text("mlflow: {tracking_uri: x}\n")

    assert config.get_app_config()["mlflow"]["tracking_uri"] == "x"
    with pytest.raises(ValueError, match="DB_NAME"):
        config.get_db_settings()
