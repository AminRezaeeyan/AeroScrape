import yaml
import os
import logging

logging.basicConfig(level = logging.INFO, format = '%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_path = "config/pipeline_config.yaml"):
    """Loads configuration from a YAML file."""
    try:
        if not os.path.exists(config_path) and 'dags' in os.getcwd():
            project_root = os.path.dirname(os.getcwd())
            config_path  = os.path.join(project_root, config_path)
        
        logger.info(f"Attempting to load config from: {os.path.abspath(config_path)}")
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info("Configuration loaded successfully.")
        return config
    except FileNotFoundError:
        logger.error(f"Configuration file not found at {config_path}. Please ensure the path is correct.")
        raise
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        raise

if __name__ == '__main__':
    # For testing the loader
    try:
        config = load_config("config/pipeline_config.yaml")
        print(config)
    except Exception as e:
        print(f"Failed to load config for test: {e}")