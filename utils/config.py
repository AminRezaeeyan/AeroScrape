import os
import yaml
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)

# This ensures environment variables are available as soon as this module is imported.
load_dotenv()

_app_config_cache = None

def _load_yaml_config(config_file_path):
    """Loads YAML configuration from the specified path."""
    try:
        with open(config_file_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logger.critical(f"YAML configuration file not found at {config_file_path}.")
        raise
    except yaml.YAMLError as e:
        logger.critical(f"Error parsing YAML configuration file {config_file_path}: {e}")
        raise
    except Exception as e:
        logger.critical(f"Unexpected error loading YAML configuration from {config_file_path}: {e}")
        raise

def get_app_config():
    """
    Returns the combined application configuration (YAML + Environment Variables).
    Loads only once and caches the result.
    """
    global _app_config_cache
    if _app_config_cache is not None:
        return _app_config_cache

    logger.info("Initializing application configuration...")
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    yaml_config_path = os.path.join(project_root, "config.yaml")
    yaml_config = _load_yaml_config(yaml_config_path)
    final_config = yaml_config if yaml_config is not None else {}

    # Database credentials (direct from environment variables)
    db_config = {
        "DB_NAME": os.getenv("DB_NAME"),
        "DB_USER": os.getenv("DB_USER"),
        "DB_PASSWORD": os.getenv("DB_PASSWORD"),
        "DB_HOST": os.getenv("DB_HOST"),
        "DB_PORT": os.getenv("DB_PORT")
    }

    # Basic validation for essential DB env vars
    for key, value in db_config.items():
        if value is None:
            logger.critical(f"Environment variable '{key}' is required but not set. Check your .env file.")
            raise ValueError(f"Missing essential environment variable: {key}")

    if 'mlflow' not in final_config or 'tracking_uri' not in final_config['mlflow']:
        logger.critical("MLflow tracking_uri is missing from config.yaml under the 'mlflow' section.")
        raise ValueError("Missing MLflow tracking_uri configuration in config.yaml")

    final_config['database'] = db_config

    _app_config_cache = final_config
    logger.info("Application configuration loaded and cached.")
    return _app_config_cache

# --- Global Access Point ---
# Call get_app_config once to populate the cache when this module is first imported.
# This ensures that any subsequent calls to get_app_config are fast.
try:
    _ = get_app_config()
except Exception as e:
    logger.critical(f"Failed to initialize global application configuration: {e}")