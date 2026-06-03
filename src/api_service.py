import pandas as pd
import mlflow
import mlflow.sklearn  # Updated: load native sklearn/lightgbm estimators
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
from datetime import datetime
from contextlib import asynccontextmanager
import joblib
import os
import logging
import sys
import shap

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.config import get_app_config

logger = logging.getLogger(__name__)

model = None
preprocessor = None
app_config = None
explainer = None  # Added: Global SHAP explainer object

def load_model_and_preprocessor():
    """
    Attempts to load the ML model from MLflow and the local preprocessor.
    Sets global variables to None if they cannot be loaded.
    """
    global model, preprocessor, app_config, explainer
    
    try:
        app_config = get_app_config()
        mlflow_settings = app_config['mlflow']
        pipeline_settings = app_config['pipeline']
        mlflow_tracking_uri = mlflow_settings['tracking_uri']
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        
        # 1. Attempt to load native ML model from MLflow Registry
        registered_model_name = pipeline_settings['model_registry_name_regression']
        model_alias = pipeline_settings['model_alias_for_api']
        model_uri = f"models:/{registered_model_name}@{model_alias}"
        
        try:
            # Loaded as a native sklearn/LGBM model to support SHAP TreeExplainer
            model = mlflow.sklearn.load_model(model_uri)
            logger.info(f"✅ Native ML model '{registered_model_name}' loaded successfully.")
            
            # Initialize global SHAP TreeExplainer
            explainer = shap.TreeExplainer(model)
            logger.info("✅ SHAP TreeExplainer initialized successfully.")
        except Exception as me:
            logger.warning(f"⚠️ Could not load ML model from registry: {me}. Prediction will be unavailable.")
            model = None
            explainer = None

        # 2. Attempt to load preprocessor locally
        preprocessor_local_path = os.path.join(
            project_root, 
            pipeline_settings['model_output_dir'], 
            pipeline_settings['preprocessor_filename']
        )
        if os.path.exists(preprocessor_local_path):
            try:
                preprocessor = joblib.load(preprocessor_local_path)
                logger.info(f"✅ Preprocessor loaded successfully from {preprocessor_local_path}.")
            except Exception as pe:
                logger.warning(f"⚠️ Could not load preprocessor from local path: {pe}. Prediction will be unavailable.")
                preprocessor = None
        else:
            logger.warning(f"⚠️ Preprocessor not found at {preprocessor_local_path}. Prediction will be unavailable.")
            preprocessor = None

    except Exception as e:
        logger.error(f"❌ Error during component initialization: {e}", exc_info=True)
        model = None
        preprocessor = None
        explainer = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Application startup: Initializing application components...")
    load_model_and_preprocessor()
    yield
    logger.info("Application shutdown: Releasing resources.")
    global model, preprocessor, explainer
    model = None
    preprocessor = None
    explainer = None

app = FastAPI(
    title="Flight Delay Prediction API",
    description="API to predict flight delay with explainable AI support.",
    version="1.1.0",
    lifespan=lifespan
)

class FlightInput(BaseModel):
    airline: str
    airport: str
    destination_or_origin: str
    aircraft: str
    scheduled_datetime: str

# Added schemas for Explainability payload
class FeatureContribution(BaseModel):
    feature: str
    contribution: float

class PredictionOutput(BaseModel):
    predicted_delay_minutes: float
    is_delayed: bool
    explainability: list[FeatureContribution]  # Added: list of top contributing features

def clean_feature_name(name: str) -> str:
    """Helper to translate internal one-hot feature names to human-readable text."""
    name = name.replace("cat__", "").replace("num__", "")
    name = name.replace("airline_", "Airline: ").replace("airport_", "Airport: ")
    name = name.replace("scheduled_season_", "Season: ").replace("scheduled_time_of_day_", "Time of Day: ")
    name = name.replace("scheduled_day_of_week_", "Day of Week: ")
    return name.replace("_", " ").title()

def create_features(data: FlightInput) -> pd.DataFrame:
    if app_config is None:
        raise RuntimeError("Application configuration not loaded.")

    df = pd.DataFrame([data.model_dump()])

    try:
        df['scheduled_datetime'] = pd.to_datetime(df['scheduled_datetime'], errors='raise')
    except Exception as e:
        logger.error(f"Error parsing scheduled_datetime '{data.scheduled_datetime}': {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid scheduled_datetime format. Use YYYY-MM-DD HH:MM:SS.")

    dt_series = df['scheduled_datetime']

    df['scheduled_day_of_week'] = dt_series.dt.day_name()
    df['is_weekend'] = df['scheduled_day_of_week'].isin(['Thursday', 'Friday']).astype(int)
    df['Scheduled_Hour_of_Day'] = dt_series.dt.hour
    df['scheduled_minute'] = dt_series.dt.minute

    def get_time_of_day(dt_obj):
        h = dt_obj.hour
        if 5 <= h <= 11: return "Morning"
        elif 12 <= h <= 16: return "Afternoon"
        elif 17 <= h <= 20: return "Evening"
        return "Night"
    df['scheduled_time_of_day'] = dt_series.apply(get_time_of_day)

    def get_season(dt_obj):
        m, d = dt_obj.month, dt_obj.day
        if (m == 3 and d >= 21) or m in [4, 5] or (m == 6 and d <= 20): return "Spring"
        if (m == 6 and d >= 21) or m in [7, 8] or (m == 9 and d <= 22): return "Summer"
        if (m == 9 and d >= 23) or m in [10, 11] or (m == 12 and d <= 21): return "Autumn"
        return "Winter"
    df['scheduled_season'] = dt_series.apply(get_season)

    nourooz_4_dates = [(3, 21), (3, 22), (3, 23), (3, 24)]
    df['is_Nourooz_4'] = dt_series.apply(lambda x: 1 if (x.month, x.day) in nourooz_4_dates else 0)
    df['is_Nourooz_13'] = dt_series.apply(lambda x: 1 if (x.month == 3 and x.day >= 21) or (x.month == 4 and x.day <= 2) else 0)

    other_fixed_holidays = [(3, 20), (2, 11), (6, 4), (6, 5)]
    def check_normal_holiday(row_dt, is_nourooz_13_val):
        if is_nourooz_13_val == 1: return 0
        if (row_dt.month, row_dt.day) in other_fixed_holidays: return 1
        return 0
    
    df['Normal_holiday'] = df.apply(lambda row: check_normal_holiday(row['scheduled_datetime'], row['is_Nourooz_13']), axis=1)

    pipeline_config = app_config['pipeline']
    numerical_features = pipeline_config['numerical_features']
    categorical_features = pipeline_config['categorical_features']

    feature_columns_for_model = numerical_features + categorical_features
    
    for col in feature_columns_for_model:
        if col not in df.columns:
            df[col] = None 
            logger.warning(f"Feature '{col}' was not generated/found; filled with None.")

    if 'scheduled_datetime' in df.columns and 'scheduled_datetime' not in feature_columns_for_model:
        df = df.drop(columns=['scheduled_datetime'])

    return df[feature_columns_for_model]

@app.post("/predict", response_model=PredictionOutput, status_code=status.HTTP_200_OK)
async def predict(flight_data: FlightInput):
    """
    Predicts flight delay and calculates individual feature contributions using SHAP.
    """
    if model is None or preprocessor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, 
            detail="Model, preprocessor, or explainability systems are currently unavailable."
        )

    try:
        features_df = create_features(flight_data)
        transformed_features = preprocessor.transform(features_df)
        
        # 1. Run prediction
        predicted_delay_minutes = float(model.predict(transformed_features)[0])
        delay_threshold = app_config['pipeline']['delay_threshold_minutes']
        is_delayed = predicted_delay_minutes > delay_threshold

        # 2. Calculate SHAP Explainability
        explainability_data = []
        if explainer is not None:
            try:
                shap_values = explainer.shap_values(transformed_features)
                # For LightGBM Regressor, shap_values is a 2D array: (n_samples, n_features)
                single_shap_values = shap_values[0]
                
                # Retrieve mapped feature column output names from preprocessor ColumnTransformer
                feature_names = preprocessor.get_feature_names_out()
                
                # Pair, clean names, and sort by absolute contribution
                raw_contributions = []
                for name, value in zip(feature_names, single_shap_values):
                    if abs(value) > 0.05:  # Filter out features with negligible contributions
                        raw_contributions.append(
                            FeatureContribution(
                                feature=clean_feature_name(name),
                                contribution=round(float(value), 3)
                            )
                        )
                explainability_data = sorted(raw_contributions, key=lambda x: abs(x.contribution), reverse=True)
            except Exception as se:
                logger.error(f"Error executing SHAP computation: {se}", exc_info=True)

        return {
            "predicted_delay_minutes": round(predicted_delay_minutes, 2),
            "is_delayed": is_delayed,
            "explainability": explainability_data
        }
    except Exception as e:
        logger.error(f"Error during prediction: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Prediction failed: {e}")

@app.post("/reload", status_code=status.HTTP_200_OK)
async def reload_assets():
    logger.info("Manual reload triggered for model and preprocessor.")
    load_model_and_preprocessor()
    if model is not None and preprocessor is not None:
        return {"status": "success", "message": "Model and preprocessor reloaded successfully."}
    else:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, 
            detail="Reload attempted but one or more components failed to load."
        )

@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    if model is not None and preprocessor is not None:
        return {"status": "ok", "message": "Model and preprocessor loaded successfully."}
    else:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, 
            detail="Model or preprocessor not loaded."
        )