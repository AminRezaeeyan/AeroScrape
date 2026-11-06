"""Application configuration: `config.yaml` for settings, environment variables for secrets."""

import logging
import os
from functools import cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

CONFIG_FILENAME = "config.yaml"
DB_ENV_VARS = ("DB_NAME", "DB_USER", "DB_PASSWORD", "DB_HOST", "DB_PORT")


@cache
def project_root() -> Path:
    """Return the directory that holds `config.yaml`.

    Resolution order: the `AEROSCRAPE_HOME` environment variable, the current
    directory or any of its parents, then the source checkout this file lives in.
    """
    env_home = os.getenv("AEROSCRAPE_HOME")
    if env_home:
        return Path(env_home).resolve()

    cwd = Path.cwd().resolve()
    for candidate in (cwd, *cwd.parents):
        if (candidate / CONFIG_FILENAME).is_file():
            return candidate

    return Path(__file__).resolve().parents[2]


def resolve_path(relative_path: str | os.PathLike[str]) -> Path:
    """Resolve a path from `config.yaml` against the project root."""
    path = Path(relative_path)
    return path if path.is_absolute() else project_root() / path


def _load_yaml_config(config_file_path: Path) -> dict[str, Any]:
    try:
        with open(config_file_path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        logger.critical("YAML configuration file not found at %s", config_file_path)
        raise
    except yaml.YAMLError:
        logger.critical("Could not parse YAML configuration file %s", config_file_path)
        raise


@cache
def get_app_config() -> dict[str, Any]:
    """Return the application configuration, loaded once and cached.

    The MLflow tracking URI from `MLFLOW_TRACKING_URI` overrides the YAML value,
    which lets the same config work on the host and inside Docker.
    """
    load_dotenv(project_root() / ".env")
    config = _load_yaml_config(project_root() / CONFIG_FILENAME)

    mlflow_config = config.setdefault("mlflow", {})
    env_mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")
    if env_mlflow_uri:
        logger.info("Using MLflow tracking URI from environment: %s", env_mlflow_uri)
        mlflow_config["tracking_uri"] = env_mlflow_uri
    if "tracking_uri" not in mlflow_config:
        raise ValueError("MLflow tracking_uri is missing from config.yaml and MLFLOW_TRACKING_URI is not set")

    logger.info("Application configuration loaded from %s", project_root() / CONFIG_FILENAME)
    return config


def get_db_settings() -> dict[str, str]:
    """Return database connection settings from the environment.

    Only code that talks to the flight database needs these, so they are checked
    here rather than when the configuration is loaded.
    """
    load_dotenv(project_root() / ".env")
    missing = [name for name in DB_ENV_VARS if not os.getenv(name)]
    if missing:
        raise ValueError(f"Missing required environment variables: {', '.join(missing)}. Check your .env file.")
    return {name: os.environ[name] for name in DB_ENV_VARS}


def apply_profile(config: dict[str, Any], profile: str | None) -> dict[str, Any]:
    """Return a copy of `config` with `profiles.<profile>` merged over it (None or "iran_fids" = no change).

    Profiles let one pipeline run on another data source, e.g. `SOURCE=bts`, without a second config file.
    """
    if not profile or profile == config.get("data_source", "iran_fids"):
        return config
    profiles = config.get("profiles", {})
    if profile not in profiles:
        raise KeyError(f"Unknown data source profile {profile!r}; defined: {sorted(profiles)}")
    return _deep_merge(config, profiles[profile]) | {"data_source": profile}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        merged[key] = (
            _deep_merge(base[key], value) if isinstance(value, dict) and isinstance(base.get(key), dict) else value
        )
    return merged
