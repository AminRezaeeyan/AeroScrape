import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if project_root not in sys.path:
    sys.path.insert(0, project_root)
    
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import json
import logging
import pandas as pd
import numpy as np
from utils.config import get_app_config

logger = logging.getLogger(__name__)

def train_regression_model(X_train, y_train):
    app_config = get_app_config()
    lgbm_params = app_config['pipeline']['lgbm_params']
    random_state = app_config['pipeline']['random_state']

    logger.info("Training regression model (LightGBM)...")
    model = lgb.LGBMRegressor(**lgbm_params, random_state=random_state)
    model.fit(X_train, y_train)
    logger.info("Regression model training complete.")
    return model

def evaluate_regression_model(model, X_test, y_test):
    app_config = get_app_config()
    accuracy_tolerance_minutes = app_config['pipeline']['delay_threshold_minutes']

    logger.info(f"Evaluating regression model. Accuracy tolerance: ±{accuracy_tolerance_minutes} minutes.")
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = mse**0.5
    r2 = r2_score(y_test, predictions)

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

def train_classification_model(X_train, y_train):
    app_config = get_app_config()
    logistic_params = app_config['pipeline']['logistic_params']
    random_state = app_config['pipeline']['random_state']

    logger.info("Training classification model (Logistic Regression)...")
    model = LogisticRegression(**logistic_params, random_state=random_state)
    model.fit(X_train, y_train)
    logger.info("Classification model training complete.")
    return model

def evaluate_classification_model(model, X_test, y_test):
    logger.info("Evaluating classification model...")
    predictions = model.predict(X_test)
    pred_proba = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, zero_division=0)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    try:
        auc = roc_auc_score(y_test, pred_proba)
    except ValueError:
        auc = float('nan')
        logger.warning("ROC AUC score could not be calculated (likely only one class in y_test).")

    metrics = {"Accuracy": accuracy, "F1_Score": f1, "Precision": precision, "Recall": recall, "AUC": auc}
    logger.info(f"Classification Metrics: {metrics}")
    return metrics, predictions

def save_model(model, model_name, subfolder=""):
    app_config = get_app_config()
    base_model_output_dir = app_config['pipeline']['model_output_dir']
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_file_dir)

    model_dir = os.path.join(project_root, base_model_output_dir, subfolder)
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, f"{model_name}.joblib")
    joblib.dump(model, model_path)
    logger.info(f"Model '{model_name}' saved to {model_path}")
    return model_path

def save_metrics(metrics, model_name, subfolder=""):
    app_config = get_app_config() 
    base_model_output_dir = app_config['pipeline']['model_output_dir'] 

    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_file_dir)
    metrics_dir = os.path.join(project_root, base_model_output_dir, subfolder)
    os.makedirs(metrics_dir, exist_ok=True)

    metrics_path = os.path.join(metrics_dir, f"{model_name}_metrics.json")

    serializable_metrics = {k: (float(v) if hasattr(v, 'item') else v) for k, v in metrics.items()}

    with open(metrics_path, 'w') as f:
        json.dump(serializable_metrics, f, indent=4)
    logger.info(f"Metrics for '{model_name}' saved to {metrics_path}")
    return metrics_path