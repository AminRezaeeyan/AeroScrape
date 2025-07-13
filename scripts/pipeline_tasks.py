import logging
import os
import pickle
import json
import mlflow

# Import all necessary functions from your other project scripts
from scripts.config_loader import load_config
from scripts.data_validator import validate_raw_data
from scripts.data_cleaner import clean_and_engineer_features
from scripts.data_processor import load_data, preprocess_data
from scripts.hyperparameter_tuner import (
    tune_regression_hyperparameters,
    tune_classification_hyperparameters,
    save_best_params
)
from scripts.model_trainer import (
    train_regression_model, evaluate_regression_model,
    train_classification_model, evaluate_classification_model,
    # save_model, save_metrics # MLflow will handle saving, or keep if redundant saving is desired
)
from .association_miner import find_association_rules

# Import save_model and save_metrics if you still want to save them outside MLflow
from scripts.model_trainer import save_model as save_model_locally 
from scripts.model_trainer import save_metrics as save_metrics_locally


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# MLflow Tracking URI - use environment variable or a default
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "file:///tmp/mlruns") # Default to local /tmp/mlruns

def get_project_root():
    """Helper function to find the project root from the script's execution context."""
    current_dir = os.getcwd()
    if 'dags' in current_dir: # Running in Airflow worker
        return os.path.dirname(current_dir)
    # Assuming script is in 'scripts' dir relative to project root for local execution
    elif os.path.basename(current_dir) == 'scripts':
        return os.path.dirname(current_dir)
    return current_dir # Fallback to current working directory

# --- TASK DEFINITIONS ---

def task_validate_raw_data(**kwargs):
    """TASK 1: Runs data validation on the raw input file."""
    logger.info("Starting task: Validate Raw Data")
    project_root = get_project_root()
    config_file_path = os.path.join(project_root, "config/pipeline_config.yaml")
    config = load_config(config_file_path)
    
    # Check for DAG param override
    raw_data_path_override = kwargs['params'].get('raw_data_path_override')
    if raw_data_path_override:
        raw_data_path = os.path.join(project_root, raw_data_path_override)
        logger.info(f"Using overridden raw_data_path from DAG params: {raw_data_path}")
    else:
        raw_data_path = os.path.join(project_root, config['raw_data_path'])
        logger.info(f"Using raw_data_path from config: {raw_data_path}")
    
    validate_raw_data(raw_data_path)
    logger.info("Raw data validation task completed successfully.")


def task_clean_and_engineer_data(**kwargs):
    """TASK 2: Runs the data cleaning and feature engineering script."""
    logger.info("Starting task: Clean and Engineer Data")
    project_root = get_project_root()
    config_file_path = os.path.join(project_root, "config/pipeline_config.yaml")
    config = load_config(config_file_path)

    # Check for DAG param override for raw_data_path
    raw_data_path_override = kwargs['params'].get('raw_data_path_override')
    if raw_data_path_override:
        raw_data_path = os.path.join(project_root, raw_data_path_override)
        logger.info(f"Using overridden raw_data_path from DAG params for cleaning: {raw_data_path}")
    else:
        raw_data_path = os.path.join(project_root, config['raw_data_path'])
        logger.info(f"Using raw_data_path from config for cleaning: {raw_data_path}")
    
    cleaned_data_path = os.path.join(project_root, config['cleaned_data_path'])
    
    os.makedirs(os.path.dirname(cleaned_data_path), exist_ok=True)
    
    clean_and_engineer_features(raw_data_path, cleaned_data_path)
    logger.info("Data cleaning and feature engineering complete.")


def task_preprocess_for_modeling(**kwargs):
    """TASK 3: Loads the CLEANED data and prepares it for modeling."""
    logger.info("Starting task: Preprocess Data for Modeling")
    project_root = get_project_root()
    ti = kwargs['ti']
    config_file_path = os.path.join(project_root, "config/pipeline_config.yaml")
    config = load_config(config_file_path)
    
    # The data_path for load_data should be the cleaned_data_path
    config['data_path'] = os.path.join(project_root, config['cleaned_data_path'])

    df = load_data(config) # load_data now expects config['data_path'] to be absolute or resolvable
    if df.empty:
        raise ValueError("Failed to load cleaned data or DataFrame is empty.")

    # Pass project_root to preprocess_data so it can construct absolute paths for saving preprocessor
    X_train_p, X_test_p, y_reg_train, y_reg_test, y_cls_train, y_cls_test, preprocessor_path = preprocess_data(df, config, project_root)
    
    processed_data_dir = os.path.join(project_root, os.path.dirname(config['processed_data_path'])) # Assuming processed_data_path is relative in config
    os.makedirs(processed_data_dir, exist_ok=True)
    
    processed_data_paths = {
        "X_train_p_path": os.path.join(processed_data_dir, "X_train_p.pkl"),
        "X_test_p_path": os.path.join(processed_data_dir, "X_test_p.pkl"),
        "y_reg_train_path": os.path.join(processed_data_dir, "y_reg_train.pkl"),
        "y_reg_test_path": os.path.join(processed_data_dir, "y_reg_test.pkl"),
        "y_cls_train_path": os.path.join(processed_data_dir, "y_cls_train.pkl"),
        "y_cls_test_path": os.path.join(processed_data_dir, "y_cls_test.pkl"),
        "preprocessor_path": preprocessor_path # Add the actual preprocessor path
    }

    for path_key, path_value in processed_data_paths.items():
        # preprocessor_path is already absolute, others need dir creation
        if path_key != "preprocessor_path":
            dir_name = os.path.dirname(path_value)
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)

    with open(processed_data_paths["X_train_p_path"], "wb") as f: pickle.dump(X_train_p, f)
    with open(processed_data_paths["X_test_p_path"], "wb") as f: pickle.dump(X_test_p, f)
    with open(processed_data_paths["y_reg_train_path"], "wb") as f: pickle.dump(y_reg_train, f)
    with open(processed_data_paths["y_reg_test_path"], "wb") as f: pickle.dump(y_reg_test, f)
    with open(processed_data_paths["y_cls_train_path"], "wb") as f: pickle.dump(y_cls_train, f)
    with open(processed_data_paths["y_cls_test_path"], "wb") as f: pickle.dump(y_cls_test, f)

    ti.xcom_push(key='processed_data_paths', value=processed_data_paths)
    logger.info(f"Data preprocessing for modeling complete. Paths pushed to XCom: {processed_data_paths}")


def task_tune_hyperparameters(**kwargs):
    """TASK 4: Runs hyperparameter tuning for both models."""
    logger.info("Starting task: Tune Hyperparameters")
    ti = kwargs['ti']
    project_root = get_project_root()
    config_file_path = os.path.join(project_root, "config/pipeline_config.yaml")
    config = load_config(config_file_path)
    
    if not config.get('tuning', {}).get('enabled', False):
        logger.info("Hyperparameter tuning is disabled in config. Skipping task.")
        ti.xcom_push(key='best_params_path', value=None) # Push None so downstream tasks know
        return

    processed_data_paths = ti.xcom_pull(task_ids='preprocess_for_modeling_task', key='processed_data_paths')
    if not processed_data_paths:
        raise ValueError("Failed to pull processed data paths from XCom for tuning.")

    with open(processed_data_paths["X_train_p_path"], "rb") as f: X_train = pickle.load(f)
    with open(processed_data_paths["X_test_p_path"], "rb") as f: X_val = pickle.load(f) # Using test set as validation for tuning
    with open(processed_data_paths["y_reg_train_path"], "rb") as f: y_reg_train = pickle.load(f)
    with open(processed_data_paths["y_reg_test_path"], "rb") as f: y_reg_val = pickle.load(f) # Using test set as validation for tuning
    with open(processed_data_paths["y_cls_train_path"], "rb") as f: y_cls_train = pickle.load(f)
    with open(processed_data_paths["y_cls_test_path"], "rb") as f: y_cls_val = pickle.load(f) # Using test set as validation for tuning

    n_trials_reg = config.get('tuning', {}).get('n_trials_regression', 50)
    best_reg_params = tune_regression_hyperparameters(X_train, y_reg_train, X_val, y_reg_val, n_trials_reg)
    
    n_trials_cls = config.get('tuning', {}).get('n_trials_classification', 25)
    best_cls_params = tune_classification_hyperparameters(X_train, y_cls_train, X_val, y_cls_val, n_trials_cls)

    all_best_params = {"regression": best_reg_params, "classification": best_cls_params}
    best_params_path = os.path.join(project_root, config['best_params_path']) # Assuming relative path in config
    os.makedirs(os.path.dirname(best_params_path), exist_ok=True)
    save_best_params(all_best_params, best_params_path)
    
    ti.xcom_push(key='best_params_path', value=best_params_path)
    logger.info(f"Hyperparameter tuning complete. Best params saved to {best_params_path} and pushed to XCom.")


def task_train_evaluate_regression(**kwargs):
    """TASK 5a: Trains, evaluates, and registers the regression model with MLflow."""
    logger.info("Starting task: Train, Evaluate, and Register Regression Model")
    ti = kwargs['ti']
    project_root = get_project_root()
    config_file_path = os.path.join(project_root, "config/pipeline_config.yaml")
    config = load_config(config_file_path)

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    logger.info(f"MLflow tracking URI set to: {MLFLOW_TRACKING_URI}")

    processed_data_paths = ti.xcom_pull(task_ids='preprocess_for_modeling_task', key='processed_data_paths')
    best_params_path = ti.xcom_pull(task_ids='tune_hyperparameters_task', key='best_params_path')

    if not processed_data_paths:
        raise ValueError("Failed to pull processed data paths from XCom.")

    with open(processed_data_paths["X_train_p_path"], "rb") as f: X_train_p = pickle.load(f)
    with open(processed_data_paths["X_test_p_path"], "rb") as f: X_test_p = pickle.load(f)
    with open(processed_data_paths["y_reg_train_path"], "rb") as f: y_reg_train = pickle.load(f)
    with open(processed_data_paths["y_reg_test_path"], "rb") as f: y_reg_test = pickle.load(f)

    preprocessor_path = processed_data_paths.get("preprocessor_path")

    mlflow.set_experiment(config.get("mlflow_experiment_name", "flight-delay-prediction"))

    with mlflow.start_run(run_name="regression_training") as run:
        logger.info(f"MLflow run started for Regression. Run ID: {run.info.run_id}")
        
        mlflow.log_artifact(config_file_path, artifact_path="config")
        logger.info(f"Logged configuration file to MLflow: {config_file_path}")

        # Log preprocessor if available
        if preprocessor_path and os.path.exists(preprocessor_path):
            mlflow.log_artifact(preprocessor_path, artifact_path="preprocessor")
            logger.info(f"Logged preprocessor to MLflow from: {preprocessor_path}")
        else:
            logger.warning(f"Preprocessor not found at {preprocessor_path} or path is None. Skipping preprocessor logging.")


        model_params = config.get('lgbm_params', {}).copy() 
        if best_params_path and os.path.exists(best_params_path):
            with open(best_params_path, 'r') as f:
                all_best_params = json.load(f)
            tuned_params = all_best_params.get('regression', {})
            model_params.update(tuned_params)
            logger.info(f"Loaded tuned parameters for regression: {tuned_params}")
        else:
            logger.info("Using default regression parameters from config (or tuning was disabled/failed).")
        
        mlflow.log_params(model_params)
        logger.info(f"Logged parameters to MLflow: {model_params}")
        
        current_run_config = config.copy()
        current_run_config['lgbm_params'] = model_params

        reg_model = train_regression_model(X_train_p, y_reg_train, current_run_config)
        
        accuracy_tolerance = config.get('regression_accuracy_tolerance_minutes', 15)
        reg_metrics, _ = evaluate_regression_model(reg_model, X_test_p, y_reg_test, accuracy_tolerance_minutes=accuracy_tolerance)
        
        serializable_metrics = {k: (float(v) if hasattr(v, 'item') else v) for k, v in reg_metrics.items()}
        mlflow.log_metrics(serializable_metrics)
        logger.info(f"Logged serializable metrics to MLflow: {serializable_metrics}")
        
        # Log the model (preprocessor is already logged as a separate artifact in this run)
        mlflow.sklearn.log_model(
            sk_model=reg_model,
            artifact_path="lgbm-regressor",
            registered_model_name="flight-delay-regressor"
        )
        logger.info("Regression model logged and registered with MLflow.")
        
    logger.info("Regression model training and MLflow logging complete.")


def task_train_evaluate_classification(**kwargs):
    """TASK 5b: Trains, evaluates, and registers the classification model with MLflow."""
    logger.info("Starting task: Train, Evaluate, and Register Classification Model")
    ti = kwargs['ti']
    project_root = get_project_root()
    config_file_path = os.path.join(project_root, "config/pipeline_config.yaml")
    config = load_config(config_file_path)

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    logger.info(f"MLflow tracking URI set to: {MLFLOW_TRACKING_URI}")

    processed_data_paths = ti.xcom_pull(task_ids='preprocess_for_modeling_task', key='processed_data_paths')
    best_params_path = ti.xcom_pull(task_ids='tune_hyperparameters_task', key='best_params_path')

    if not processed_data_paths:
        raise ValueError("Failed to pull processed data paths from XCom for classification task.")

    with open(processed_data_paths["X_train_p_path"], "rb") as f: X_train_p = pickle.load(f)
    with open(processed_data_paths["X_test_p_path"], "rb") as f: X_test_p = pickle.load(f)
    with open(processed_data_paths["y_cls_train_path"], "rb") as f: y_cls_train = pickle.load(f)
    with open(processed_data_paths["y_cls_test_path"], "rb") as f: y_cls_test = pickle.load(f)

    preprocessor_path = processed_data_paths.get("preprocessor_path")

    mlflow.set_experiment(config.get("mlflow_experiment_name", "flight-delay-prediction"))

    with mlflow.start_run(run_name="classification_training") as run:
        logger.info(f"MLflow run started for Classification. Run ID: {run.info.run_id}")

        mlflow.log_artifact(config_file_path, artifact_path="config")
        logger.info(f"Logged configuration file to MLflow: {config_file_path}")

        # Log preprocessor if available
        if preprocessor_path and os.path.exists(preprocessor_path):
            mlflow.log_artifact(preprocessor_path, artifact_path="preprocessor")
            logger.info(f"Logged preprocessor to MLflow from: {preprocessor_path}")
        else:
            logger.warning(f"Preprocessor not found at {preprocessor_path} or path is None. Skipping preprocessor logging.")
        
        model_params = config.get('logistic_params', {}).copy()
        if best_params_path and os.path.exists(best_params_path):
            with open(best_params_path, 'r') as f:
                all_best_params = json.load(f)
            tuned_params = all_best_params.get('classification', {})
            model_params.update(tuned_params)
            logger.info(f"Loaded tuned parameters for classification: {tuned_params}")
        else:
            logger.info("Using default classification parameters from config (or tuning was disabled/failed).")

        mlflow.log_params(model_params)
        logger.info(f"Logged classification parameters to MLflow: {model_params}")

        current_run_config = config.copy()
        current_run_config['logistic_params'] = model_params

        cls_model = train_classification_model(X_train_p, y_cls_train, current_run_config)
        cls_metrics, _ = evaluate_classification_model(cls_model, X_test_p, y_cls_test)
        
        serializable_cls_metrics = {k: (float(v) if hasattr(v, 'item') else v) for k, v in cls_metrics.items()}
        mlflow.log_metrics(serializable_cls_metrics)
        logger.info(f"Logged serializable classification metrics to MLflow: {serializable_cls_metrics}")

        # Log the model (preprocessor is already logged as a separate artifact in this run)
        mlflow.sklearn.log_model(
            sk_model=cls_model,
            artifact_path="logistic-classifier",
            registered_model_name="flight-delay-classifier"
        )
        logger.info("Classification model logged and registered with MLflow.")

    logger.info("Classification model training and MLflow logging complete.")

def task_run_association_mining():
    """
    Airflow task to run the FP-Growth association Rule mining Task
    """
    logging.info("--- Starting Association Rule Mining Task ---")
    config = load_config()

    cleaned_data_path = os.path.join(config['data']['cleaned']['path'], config['data']['cleaned']['filename'])
    output_path = os.path.join(config['data']['results']['path'], config['data']['results']['association_rules_filename'])
    
    # Create results directory if it doesn't exist
    os.makedirs(config['data']['results']['path'], exist_ok=True)
    
    find_association_rules(
        data_path=cleaned_data_path,
        output_path=output_path,
        min_support=config['association_mining']['min_support'],
        min_threshold=config['association_mining']['min_threshold']
    )
    logging.info("--- Association Rule Mining Task Completed ---")




def task_cleanup_processed_data(**kwargs):
    """TASK 6: Cleans up intermediate processed data files (pickled data, not the preprocessor)."""
    logger.info("Starting task: Cleanup Processed Data")
    ti = kwargs['ti']
    processed_data_paths = ti.xcom_pull(task_ids='preprocess_for_modeling_task', key='processed_data_paths')
    if not processed_data_paths:
        logger.warning("No processed data paths found in XCom for cleanup.")
        return

    # Exclude the preprocessor and best_params from auto-cleanup as they are final artifacts or handled elsewhere
    # The preprocessor itself is an output artifact. Best_params is also an output.
    # The pickled data files (X_train, y_train etc.) are intermediate.
    paths_to_delete = {
        k: v for k, v in processed_data_paths.items() 
        if k not in ["preprocessor_path"] # Keep preprocessor, it's an artifact
    }
    # Also, don't delete best_params_path if it was pushed by the tuning task.
    # However, this task only pulls from 'preprocess_for_modeling_task' for its list of files.

    for path_key, path_value in paths_to_delete.items():
        if path_value and os.path.exists(path_value): # Check if path_value is not None
            try:
                os.remove(path_value)
                logger.info(f"Removed intermediate file: {path_value}")
            except Exception as e:
                logger.error(f"Error removing file {path_value}: {e}")
        elif path_value:
            logger.info(f"File not found for cleanup (already deleted or never created): {path_value}")

    logger.info("Processed data cleanup complete.")