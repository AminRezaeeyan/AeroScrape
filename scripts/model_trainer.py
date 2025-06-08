# scripts/model_trainer.py
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os
import json
import logging
import pandas as pd 
import numpy as np

logger = logging.getLogger(__name__)

def train_regression_model(X_train, y_train, config):
    """Trains a LightGBM regression model."""
    logger.info("Training regression model (LightGBM)...")
    model = lgb.LGBMRegressor(**config['lgbm_params'], random_state=config['random_state'])
    model.fit(X_train, y_train)
    logger.info("Regression model training complete.")
    return model

def evaluate_regression_model(model, X_test, y_test, accuracy_tolerance_minutes=15):
    """Evaluates the regression model and includes a custom accuracy metric."""
    logger.info(f"Evaluating regression model. Accuracy tolerance: ±{accuracy_tolerance_minutes} minutes.")
    predictions = model.predict(X_test)

    # Standard regression metrics
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = mse**0.5
    r2 = r2_score(y_test, predictions)

    # Custom: Calculate accuracy within the specified tolerance
    y_test_np = np.array(y_test)
    predictions_np = np.array(predictions)
    
    correct_predictions = np.sum(np.abs(y_test_np - predictions_np) <= accuracy_tolerance_minutes)
    custom_accuracy = correct_predictions / len(y_test_np) if len(y_test_np) > 0 else 0

    metrics = {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2_Score": r2,
        f"Accuracy_within_{accuracy_tolerance_minutes}min": custom_accuracy
    }
    logger.info(f"Regression Metrics: {metrics}")
    return metrics, predictions

def train_classification_model(X_train, y_train, config):
    """Trains a Logistic Regression classification model."""
    logger.info("Training classification model (Logistic Regression)...")
    model = LogisticRegression(**config['logistic_params'], random_state=config['random_state'])
    model.fit(X_train, y_train)
    logger.info("Classification model training complete.")
    return model

def evaluate_classification_model(model, X_test, y_test):
    """Evaluates the classification model."""
    logger.info("Evaluating classification model...")
    predictions = model.predict(X_test)
    pred_proba = model.predict_proba(X_test)[:, 1] # Probability of positive class

    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, zero_division=0)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    try: # AUC might fail if only one class present in y_test (should not happen with proper split)
        auc = roc_auc_score(y_test, pred_proba)
    except ValueError:
        auc = float('nan')
        logger.warning("ROC AUC score could not be calculated (likely only one class in y_test).")


    metrics = {"Accuracy": accuracy, "F1_Score": f1, "Precision": precision, "Recall": recall, "AUC": auc}
    logger.info(f"Classification Metrics: {metrics}")
    return metrics, predictions

def save_model(model, model_name, config, subfolder=""):
    """Saves the trained model and its metrics."""
    # Adjust path for Airflow context
    base_model_output_dir = config['model_output_dir']
    if not os.path.exists(base_model_output_dir) and 'dags' in os.getcwd():
        project_root = os.path.dirname(os.getcwd())
        base_model_output_dir = os.path.join(project_root, base_model_output_dir)

    model_dir = os.path.join(base_model_output_dir, subfolder)
    os.makedirs(model_dir, exist_ok=True)
    
    model_path = os.path.join(model_dir, f"{model_name}.joblib")
    joblib.dump(model, model_path)
    logger.info(f"Model '{model_name}' saved to {model_path}")
    return model_path

def save_metrics(metrics, model_name, config, subfolder=""):
    """Saves model metrics to a JSON file."""
    base_model_output_dir = config['model_output_dir']
    if not os.path.exists(base_model_output_dir) and 'dags' in os.getcwd():
        project_root = os.path.dirname(os.getcwd())
        base_model_output_dir = os.path.join(project_root, base_model_output_dir)
        
    metrics_dir = os.path.join(base_model_output_dir, subfolder)
    os.makedirs(metrics_dir, exist_ok=True)

    metrics_path = os.path.join(metrics_dir, f"{model_name}_metrics.json")
    # Convert metrics to a serializable format (e.g., handle numpy types)
    serializable_metrics = {k: (float(v) if hasattr(v, 'item') else v) for k, v in metrics.items()}

    with open(metrics_path, 'w') as f:
        json.dump(serializable_metrics, f, indent=4)
    logger.info(f"Metrics for '{model_name}' saved to {metrics_path}")
    return metrics_path

# if __name__ == '__main__':
#     # For testing trainer - Requires processed data
#     # This part is more complex to test standalone without the full pipeline context
#     # You would typically mock the input data (X_train, y_train etc.)
#     logger.info("Model trainer script can be tested by calling its functions with appropriate data.")
#     # Example:
#     from scripts.config_loader import load_config
#     test_config = load_config("../config/pipeline_config.yaml") # adjust path
#     # Assume X_train_p, y_cls_tr etc. are loaded/mocked
#     if 'X_train_p' in locals(): # Check if data_processor was run and populated these
#         cls_model = train_classification_model(X_train_p, y_cls_tr, test_config)
#         cls_metrics, _ = evaluate_classification_model(cls_model, X_test_p, y_cls_te)
#         save_model(cls_model, "logistic_regression_test", test_config, "classification")
#         save_metrics(cls_metrics, "logistic_regression_test", test_config, "classification")
#     else:
#         logger.warning("Skipping model trainer test as processed data is not available in this scope.")