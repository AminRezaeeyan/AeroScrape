import pandas as pd
import mlflow
import mlflow.pyfunc
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
from datetime import datetime, timedelta # Import timedelta for holiday logic
from contextlib import asynccontextmanager
import joblib
import os
import logging

import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.config import get_app_config

logger = logging.getLogger(__name__)

model = None
preprocessor = None
app_config = None # Global to store loaded configuration for reuse

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Context manager for application startup and shutdown events.
    Loads the ML model and preprocessor at startup from MLflow Model Registry.
    """
    global model, preprocessor, app_config

    logger.info("Application startup: Loading configuration, model, and preprocessor from MLflow Registry...")
    try:
        app_config = get_app_config()
        mlflow_settings = app_config['mlflow']
        pipeline_settings = app_config['pipeline']
        mlflow_tracking_uri = mlflow_settings['tracking_uri']
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        logger.info(f"MLflow tracking URI set to: {mlflow_tracking_uri}.")

        # Load the ML model from MLflow Model Registry
        registered_model_name = pipeline_settings['model_registry_name_regression'] # Assuming regression model for API
        model_stage = pipeline_settings['model_stage_for_api']
        model_uri = f"models:/{registered_model_name}/{model_stage}"

        model = mlflow.pyfunc.load_model(model_uri)
        logger.info(f"✅ ML model '{registered_model_name}' in stage '{model_stage}' loaded successfully from {model_uri}.")

        preprocessor_local_path = os.path.join(project_root, pipeline_settings['model_output_dir'], pipeline_settings['preprocessor_filename'])
        if not os.path.exists(preprocessor_local_path):
            raise FileNotFoundError(f"Preprocessor not found at: {preprocessor_local_path}")
        preprocessor = joblib.load(preprocessor_local_path)
        logger.info(f"✅ Preprocessor loaded successfully from {preprocessor_local_path}.")

    except FileNotFoundError as fnfe:
        logger.critical(f"🛑 Error during application startup: Essential file missing - {fnfe}", exc_info=True)
        model = None
        preprocessor = None
        raise RuntimeError(f"Failed to load essential components: {fnfe}")
    except mlflow.exceptions.MlflowException as me:
        logger.critical(f"🛑 Error during MLflow model loading: {me}", exc_info=True)
        model = None
        preprocessor = None
        raise RuntimeError(f"Failed to load MLflow model: {me}")
    except Exception as e:
        logger.critical(f"🛑 General error during application startup (model or preprocessor loading): {e}", exc_info=True)
        model = None
        preprocessor = None
        raise RuntimeError(f"Failed to load essential components: {e}")

    yield

    # Application shutdown logic
    logger.info("Application shutdown: Releasing resources.")
    model = None
    preprocessor = None

# Initialize FastAPI app with the lifespan context manager
app = FastAPI(
    title="Flight Delay Prediction API",
    description="API to predict flight delay using a machine learning model.",
    version="1.0.0",
    lifespan=lifespan
)

class FlightInput(BaseModel):
    airline: str
    airport: str
    destination_or_origin: str
    aircraft: str
    scheduled_datetime: str

class PredictionOutput(BaseModel):
    predicted_delay_minutes: float
    is_delayed: bool

def create_features(data: FlightInput) -> pd.DataFrame:
    """
    Transforms raw input data from the API request into features
    expected by the preprocessor and model.
    This logic accurately mirrors data_cleaner.py and data_processor.py's feature engineering
    using the provided scheduled_datetime.
    """
    if app_config is None:
        raise RuntimeError("Application configuration not loaded.")

    df = pd.DataFrame([data.model_dump()])

    try:
        df['scheduled_datetime'] = pd.to_datetime(df['scheduled_datetime'], errors='raise')
    except Exception as e:
        logger.error(f"Error parsing scheduled_datetime '{data.scheduled_datetime}': {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid scheduled_datetime format. Use YYYY-MM-DD HH:MM:SS.")

    dt_series = df['scheduled_datetime']

    # --- Feature Engineering mirroring data_cleaner.py ---

    # 1. Extract scheduled_day_of_week and is_weekend
    df['scheduled_day_of_week'] = dt_series.dt.day_name()
    df['is_weekend'] = df['scheduled_day_of_week'].isin(['Thursday', 'Friday']).astype(int)

    # 2. Extract Scheduled_Hour_of_Day and scheduled_minute
    df['Scheduled_Hour_of_Day'] = dt_series.dt.hour
    df['scheduled_minute'] = dt_series.dt.minute

    # 3. Determine scheduled_time_of_day
    def get_time_of_day(dt_obj):
        h = dt_obj.hour
        if 5 <= h <= 11: return "Morning"
        elif 12 <= h <= 16: return "Afternoon"
        elif 17 <= h <= 20: return "Evening"
        return "Night"
    df['scheduled_time_of_day'] = dt_series.apply(get_time_of_day)

    # 4. Determine scheduled_season
    def get_season(dt_obj):
        m, d = dt_obj.month, dt_obj.day
        if (m == 3 and d >= 21) or m in [4, 5] or (m == 6 and d <= 20): return "Spring"
        if (m == 6 and d >= 21) or m in [7, 8] or (m == 9 and d <= 22): return "Summer"
        if (m == 9 and d >= 23) or m in [10, 11] or (m == 12 and d <= 21): return "Autumn"
        return "Winter"
    df['scheduled_season'] = dt_series.apply(get_season)

    # 5. Holiday Features (mirroring data_cleaner.py - Gregorian dates)
    nourooz_4_dates = [(3, 21), (3, 22), (3, 23), (3, 24)]
    df['is_Nourooz_4'] = dt_series.apply(lambda x: 1 if (x.month, x.day) in nourooz_4_dates else 0)

    df['is_Nourooz_13'] = dt_series.apply(lambda x: 1 if (x.month == 3 and x.day >= 21) or (x.month == 4 and x.day <= 2) else 0)

    other_fixed_holidays = [(3, 20), (2, 11), (6, 4), (6, 5)]
    def check_normal_holiday(row_dt, is_nourooz_13_val):
        if is_nourooz_13_val == 1: return 0 # If it's Nourooz_13 period, it's not a 'Normal_holiday'
        if (row_dt.month, row_dt.day) in other_fixed_holidays: return 1
        return 0
    
    df['Normal_holiday'] = df.apply(lambda row: check_normal_holiday(row['scheduled_datetime'], row['is_Nourooz_13']), axis=1)

    # --- Select final features for the model ---
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
    Predicts flight delay based on input features.
    """
    if model is None or preprocessor is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model or preprocessor not loaded. Server is not ready.")

    try:
        features_df = create_features(flight_data)
        transformed_features = preprocessor.transform(features_df)
        predicted_delay_minutes = float(model.predict(transformed_features)[0])

        delay_threshold = app_config['pipeline']['delay_threshold_minutes']
        is_delayed = predicted_delay_minutes > delay_threshold

        return {
            "predicted_delay_minutes": round(predicted_delay_minutes, 2),
            "is_delayed": is_delayed
        }
    except Exception as e:
        logger.error(f"Error during prediction: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Prediction failed: {e}")

@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    """
    Health check endpoint to verify if the service is running and models are loaded.
    """
    if model is not None and preprocessor is not None:
        return {"status": "ok", "message": "Model and preprocessor loaded successfully."}
    else:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model or preprocessor not loaded.")