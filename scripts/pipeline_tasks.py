import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if project_root not in sys.path:
    sys.path.insert(0, project_root)
    
import logging
import pickle
import json
import mlflow

from utils.config import get_app_config

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
    save_model, save_metrics
)


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_project_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def task_validate_raw_data(**kwargs):
    logger.info("Starting task: Validate Raw Data")
    project_root = get_project_root()
    app_config = get_app_config()

    raw_data_path_override = kwargs['params'].get('raw_data_path_override')
    if raw_data_path_override:
        raw_data_path = os.path.join(project_root, raw_data_path_override)
        logger.info(f"Using overridden raw_data_path from DAG params: {raw_data_path}")
    else:
        raw_data_path = os.path.join(project_root, app_config['data_paths']['raw_data'])
        logger.info(f"Using raw_data_path from config: {raw_data_path}")

    validate_raw_data(raw_data_path)
    logger.info("Raw data validation task completed successfully.")


def task_clean_and_engineer_data(**kwargs):
    logger.info("Starting task: Clean and Engineer Data")
    project_root = get_project_root()
    app_config = get_app_config()

    raw_data_path_override = kwargs['params'].get('raw_data_path_override')
    if raw_data_path_override:
        raw_data_path = os.path.join(project_root, raw_data_path_override)
        logger.info(f"Using overridden raw_data_path from DAG params for cleaning: {raw_data_path}")
    else:
        raw_data_path = os.path.join(project_root, app_config['data_paths']['raw_data'])
        logger.info(f"Using raw_data_path from config for cleaning: {raw_data_path}")

    cleaned_data_path = os.path.join(project_root, app_config['data_paths']['cleaned_data'])
    os.makedirs(os.path.dirname(cleaned_data_path), exist_ok=True)

    clean_and_engineer_features(raw_data_path, cleaned_data_path)
    logger.info("Data cleaning and feature engineering complete.")


def task_preprocess_for_modeling(**kwargs):
    logger.info("Starting task: Preprocess Data for Modeling")
    ti = kwargs['ti']
    project_root = get_project_root()
    app_config = get_app_config()

    data_path_for_processor = os.path.join(project_root, app_config['data_paths']['cleaned_data'])

    df = load_data(data_path_for_processor) 
    if df.empty:
        raise ValueError("Failed to load cleaned data or DataFrame is empty.")

    X_train_p, X_test_p, y_reg_train, y_reg_test, y_cls_train, y_cls_test, preprocessor_path = \
        preprocess_data(df, project_root) 

    # Save processed data as pickle files
    processed_data_dir = os.path.join(project_root, app_config['data_paths']['processed_data_dir'])
    os.makedirs(processed_data_dir, exist_ok=True)

    processed_data_paths = {
        "X_train_p_path": os.path.join(processed_data_dir, "X_train_p.pkl"),
        "X_test_p_path": os.path.join(processed_data_dir, "X_test_p.pkl"),
        "y_reg_train_path": os.path.join(processed_data_dir, "y_reg_train.pkl"),
        "y_reg_test_path": os.path.join(processed_data_dir, "y_reg_test.pkl"),
        "y_cls_train_path": os.path.join(processed_data_dir, "y_cls_train.pkl"),
        "y_cls_test_path": os.path.join(processed_data_dir, "y_cls_test.pkl"),
        "preprocessor_path": preprocessor_path
    }

    for path_key, path_value in processed_data_paths.items():
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
    logger.info("Starting task: Tune Hyperparameters")
    ti = kwargs['ti']
    project_root = get_project_root()
    app_config = get_app_config()

    tuning_enabled = app_config['pipeline']['tuning']['enabled']
    if not tuning_enabled:
        logger.info("Hyperparameter tuning is disabled in config. Skipping task.")
        ti.xcom_push(key='best_params_path', value=None)
        return

    processed_data_paths = ti.xcom_pull(task_ids='preprocess_for_modeling_task', key='processed_data_paths')
    if not processed_data_paths:
        raise ValueError("Failed to pull processed data paths from XCom for tuning.")

    # Load preprocessed data for tuning
    with open(processed_data_paths["X_train_p_path"], "rb") as f: X_train = pickle.load(f)
    with open(processed_data_paths["X_test_p_path"], "rb") as f: X_val = pickle.load(f)
    with open(processed_data_paths["y_reg_train_path"], "rb") as f: y_reg_train = pickle.load(f)
    with open(processed_data_paths["y_reg_test_path"], "rb") as f: y_reg_val = pickle.load(f)
    with open(processed_data_paths["y_cls_train_path"], "rb") as f: y_cls_train = pickle.load(f)
    with open(processed_data_paths["y_cls_test_path"], "rb") as f: y_cls_val = pickle.load(f)

    # Get tuning parameters
    n_trials_reg = app_config['pipeline']['tuning']['n_trials_regression']
    n_trials_cls = app_config['pipeline']['tuning']['n_trials_classification']

    # Perform tuning
    best_reg_params = tune_regression_hyperparameters(X_train, y_reg_train, X_val, y_reg_val, n_trials_reg)
    best_cls_params = tune_classification_hyperparameters(X_train, y_cls_train, X_val, y_cls_val, n_trials_cls)

    all_best_params = {"regression": best_reg_params, "classification": best_cls_params}

    # Define path for temporary local storage of best params
    best_params_local_path = os.path.join(project_root, app_config['data_paths']['best_params'])

    save_best_params(all_best_params, best_params_local_path)

    # MLflow tracking for the tuning run
    mlflow_tracking_uri = app_config['mlflow']['tracking_uri']
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(app_config['mlflow']['experiment_name'])

    with mlflow.start_run(run_name="hyperparameter_tuning") as run:
        logger.info(f"MLflow run started for Hyperparameter Tuning. Run ID: {run.info.run_id}")
        mlflow.log_params({"reg_n_trials": n_trials_reg, "cls_n_trials": n_trials_cls})
        mlflow.log_params({"best_reg_params": best_reg_params})
        mlflow.log_params({"best_cls_params": best_cls_params})

        # Log the best_params.json file as an artifact of the tuning run
        mlflow.log_artifact(best_params_local_path, artifact_path="best_params")
        logger.info(f"Logged best_params.json as artifact to MLflow: {best_params_local_path}")


    ti.xcom_push(key='best_params_path', value=best_params_local_path) 
    logger.info(f"Hyperparameter tuning complete. Best params saved to {best_params_local_path} and pushed to XCom.")


def task_train_evaluate_regression(**kwargs):
    logger.info("Starting task: Train, Evaluate, and Register Regression Model")
    ti = kwargs['ti']
    project_root = get_project_root()
    app_config = get_app_config()

    mlflow_tracking_uri = app_config['mlflow']['tracking_uri'] # NEW: Read from 'mlflow' section
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(app_config['mlflow']['experiment_name'])

    processed_data_paths = ti.xcom_pull(task_ids='preprocess_for_modeling_task', key='processed_data_paths')
    best_params_local_path = ti.xcom_pull(task_ids='tune_hyperparameters_task', key='best_params_path')

    if not processed_data_paths:
        raise ValueError("Failed to pull processed data paths from XCom.")

    # Load preprocessed data
    with open(processed_data_paths["X_train_p_path"], "rb") as f: X_train_p = pickle.load(f)
    with open(processed_data_paths["X_test_p_path"], "rb") as f: X_test_p = pickle.load(f)
    with open(processed_data_paths["y_reg_train_path"], "rb") as f: y_reg_train = pickle.load(f)
    with open(processed_data_paths["y_reg_test_path"], "rb") as f: y_reg_test = pickle.load(f)

    preprocessor_path = processed_data_paths.get("preprocessor_path")

    mlflow.set_experiment(app_config['mlflow']['experiment_name'])

    with mlflow.start_run(run_name="regression_training") as run:
        logger.info(f"MLflow run started for Regression. Run ID: {run.info.run_id}")

        config_file_path = os.path.join(project_root, "config.yaml")
        mlflow.log_artifact(config_file_path, artifact_path="config")
        logger.info(f"Logged configuration file to MLflow: {config_file_path}")

        if preprocessor_path and os.path.exists(preprocessor_path):
            mlflow.log_artifact(preprocessor_path, artifact_path="preprocessor")
            logger.info(f"Logged preprocessor to MLflow from: {preprocessor_path}")
        else:
            logger.warning(f"Preprocessor not found at {preprocessor_path} or path is None. Skipping preprocessor logging.")

        model_params = app_config['pipeline']['lgbm_params'].copy()
        if best_params_local_path and os.path.exists(best_params_local_path):
            with open(best_params_local_path, 'r') as f:
                all_best_params = json.load(f)
            tuned_params = all_best_params.get('regression', {})
            model_params.update(tuned_params)
            logger.info(f"Loaded tuned parameters for regression: {tuned_params}")
        else:
            logger.info("Using default regression parameters from config (or tuning was disabled/failed).")

        mlflow.log_params(model_params)
        logger.info(f"Logged parameters to MLflow: {model_params}")

        reg_model = train_regression_model(X_train_p, y_reg_train) # Pass data only
        reg_metrics, _ = evaluate_regression_model(reg_model, X_test_p, y_reg_test) # Pass data only

        serializable_metrics = {k: (float(v) if hasattr(v, 'item') else v) for k, v in reg_metrics.items()}
        mlflow.log_metrics(serializable_metrics)
        logger.info(f"Logged serializable metrics to MLflow: {serializable_metrics}")

        mlflow.sklearn.log_model(
            sk_model=reg_model,
            artifact_path="lgbm-regressor",
            registered_model_name="flight-delay-regressor"
        )
        logger.info("Regression model logged and registered with MLflow.")

    save_metrics(reg_metrics, "lgbm_regressor", subfolder="regression")
    save_model(reg_model, "lgbm_regressor", subfolder="regression")
    logger.info("Regression model training and MLflow logging complete.")


def task_train_evaluate_classification(**kwargs):
    logger.info("Starting task: Train, Evaluate, and Register Classification Model")
    ti = kwargs['ti']
    project_root = get_project_root()
    app_config = get_app_config()

    mlflow_tracking_uri = app_config['mlflow']['tracking_uri'] # NEW: Read from 'mlflow' section
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(app_config['mlflow']['experiment_name'])

    processed_data_paths = ti.xcom_pull(task_ids='preprocess_for_modeling_task', key='processed_data_paths')
    best_params_local_path = ti.xcom_pull(task_ids='tune_hyperparameters_task', key='best_params_path')

    if not processed_data_paths:
        raise ValueError("Failed to pull processed data paths from XCom for classification task.")

    with open(processed_data_paths["X_train_p_path"], "rb") as f: X_train_p = pickle.load(f)
    with open(processed_data_paths["X_test_p_path"], "rb") as f: X_test_p = pickle.load(f)
    with open(processed_data_paths["y_cls_train_path"], "rb") as f: y_cls_train = pickle.load(f)
    with open(processed_data_paths["y_cls_test_path"], "rb") as f: y_cls_test = pickle.load(f)

    preprocessor_path = processed_data_paths.get("preprocessor_path")

    mlflow.set_experiment(app_config['mlflow']['experiment_name'])

    with mlflow.start_run(run_name="classification_training") as run:
        logger.info(f"MLflow run started for Classification. Run ID: {run.info.run_id}")

        config_file_path = os.path.join(project_root, "config.yaml")
        mlflow.log_artifact(config_file_path, artifact_path="config")
        logger.info(f"Logged configuration file to MLflow: {config_file_path}")

        if preprocessor_path and os.path.exists(preprocessor_path):
            mlflow.log_artifact(preprocessor_path, artifact_path="preprocessor")
            logger.info(f"Logged preprocessor to MLflow from: {preprocessor_path}")
        else:
            logger.warning(f"Preprocessor not found at {preprocessor_path} or path is None. Skipping preprocessor logging.")

        model_params = app_config['pipeline']['logistic_params'].copy()
        if best_params_local_path and os.path.exists(best_params_local_path):
            with open(best_params_local_path, 'r') as f:
                all_best_params = json.load(f)
            tuned_params = all_best_params.get('classification', {})
            model_params.update(tuned_params)
            logger.info(f"Loaded tuned parameters for classification: {tuned_params}")
        else:
            logger.info("Using default classification parameters from config (or tuning was disabled/failed).")

        mlflow.log_params(model_params)
        logger.info(f"Logged classification parameters to MLflow: {model_params}")

        cls_model = train_classification_model(X_train_p, y_cls_train)
        cls_metrics, _ = evaluate_classification_model(cls_model, X_test_p, y_cls_test)

        serializable_cls_metrics = {k: (float(v) if hasattr(v, 'item') else v) for k, v in cls_metrics.items()}
        mlflow.log_metrics(serializable_cls_metrics)
        logger.info(f"Logged serializable classification metrics to MLflow: {serializable_cls_metrics}")

        mlflow.sklearn.log_model(
            sk_model=cls_model,
            artifact_path="logistic-classifier",
            registered_model_name="flight-delay-classifier"
        )
        logger.info("Classification model logged and registered with MLflow.")

    save_metrics(cls_metrics, "logistic_classifier", subfolder="classification")
    save_model(cls_model, "logistic_classifier", subfolder="classification")
    logger.info("Classification model training and MLflow logging complete.")


def task_cleanup_processed_data(**kwargs):
    logger.info("Starting task: Cleanup Processed Data")
    ti = kwargs['ti']
    project_root = get_project_root()
    app_config = get_app_config()

    processed_data_paths = ti.xcom_pull(task_ids='preprocess_for_modeling_task', key='processed_data_paths')
    if not processed_data_paths:
        logger.warning("No processed data paths found in XCom for cleanup.")
        return

    # Paths to delete (all pickle files from processed_data_dir, and best_params.json)
    paths_to_delete = [
        processed_data_paths["X_train_p_path"],
        processed_data_paths["X_test_p_path"],
        processed_data_paths["y_reg_train_path"],
        processed_data_paths["y_reg_test_path"],
        processed_data_paths["y_cls_train_path"],
        processed_data_paths["y_cls_test_path"],
        os.path.join(project_root, app_config['data_paths']['best_params'])
    ]

    for path_value in paths_to_delete:
        if path_value and os.path.exists(path_value):
            try:
                os.remove(path_value)
                logger.info(f"Removed intermediate file: {path_value}")
            except Exception as e:
                logger.error(f"Error removing file {path_value}: {e}")
        elif path_value:
            logger.info(f"File not found for cleanup (already deleted or never created): {path_value}")

    # The preprocessor.joblib should typically NOT be deleted here as it's used for inference.
    # It resides in models/ which is explicitly outside data/processed/

    logger.info("Processed data cleanup complete.")