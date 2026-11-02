import logging
import os
from contextlib import asynccontextmanager

import joblib
import mlflow
import mlflow.sklearn  # Updated: load native sklearn/lightgbm estimators
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel

from aeroscrape.config import get_app_config, resolve_path
from aeroscrape.data.clean import normalize_persian
from aeroscrape.features.build import add_calendar_features
from aeroscrape.features.preprocess import variant_feature_set

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
        mlflow_settings = app_config["mlflow"]
        pipeline_settings = app_config["pipeline"]
        mlflow_tracking_uri = mlflow_settings["tracking_uri"]
        mlflow.set_tracking_uri(mlflow_tracking_uri)

        # 1. Attempt to load native ML model from MLflow Registry
        registered_model_name = pipeline_settings["model_registry_name_regression"]
        model_alias = pipeline_settings["model_alias_for_api"]
        model_uri = f"models:/{registered_model_name}@{model_alias}"

        try:
            # Loaded as a native sklearn/LGBM model to support SHAP TreeExplainer
            model = mlflow.sklearn.load_model(model_uri)
            logger.info(f"Native ML model '{registered_model_name}' loaded successfully.")

            # Initialize global SHAP TreeExplainer
            explainer = shap.TreeExplainer(model)
            logger.info("SHAP TreeExplainer initialized successfully.")
        except Exception as me:
            logger.warning(f"Could not load ML model from registry: {me}. Prediction will be unavailable.")
            model = None
            explainer = None

        # 2. Attempt to load preprocessor locally
        preprocessor_local_path = os.path.join(
            resolve_path(pipeline_settings["model_output_dir"]), pipeline_settings["preprocessor_filename"]
        )
        if os.path.exists(preprocessor_local_path):
            try:
                preprocessor = joblib.load(preprocessor_local_path)
                logger.info(f"Preprocessor loaded successfully from {preprocessor_local_path}.")
            except Exception as pe:
                logger.warning(f"Could not load preprocessor from local path: {pe}. Prediction will be unavailable.")
                preprocessor = None
        else:
            logger.warning(f"Preprocessor not found at {preprocessor_local_path}. Prediction will be unavailable.")
            preprocessor = None

    except Exception as e:
        logger.error(f"Error during component initialization: {e}", exc_info=True)
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
    lifespan=lifespan,
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
        df["scheduled_datetime"] = pd.to_datetime(df["scheduled_datetime"], errors="raise")
    except Exception as e:
        logger.error(f"Error parsing scheduled_datetime '{data.scheduled_datetime}': {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid scheduled_datetime format. Use YYYY-MM-DD HH:MM:SS.",
        ) from e

    # Same text normalization as the cleaner, so Arabic and Persian spellings match training
    for col in ("airline", "airport", "destination_or_origin", "aircraft"):
        df[col] = df[col].map(normalize_persian)
    df = add_calendar_features(df)

    # Features that need other flights (rotation, congestion, rates) can't be computed from
    # one request yet; they stay missing and the model's imputer fills them. Phase 4 adds a
    # feature lookup so the API builds them with the same code as training.
    columns = variant_feature_set(app_config, app_config["features"]["served_variant"]).columns
    missing = [col for col in columns if col not in df.columns]
    if missing:
        logger.info("Features without request-time data, left missing: %s", missing)
    return df.reindex(columns=columns)


@app.post("/predict", response_model=PredictionOutput, status_code=status.HTTP_200_OK)
async def predict(flight_data: FlightInput):
    """
    Predicts flight delay and calculates individual feature contributions using SHAP.
    """
    if model is None or preprocessor is None or app_config is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model, preprocessor, or explainability systems are currently unavailable.",
        )

    try:
        features_df = create_features(flight_data)
        transformed_features = preprocessor.transform(features_df)

        # 1. Run prediction
        predicted_delay_minutes = float(model.predict(transformed_features)[0])
        delay_threshold = app_config["pipeline"]["delay_threshold_minutes"]
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
                for name, value in zip(feature_names, single_shap_values, strict=True):
                    if abs(value) > 0.05:  # Filter out features with negligible contributions
                        raw_contributions.append(
                            FeatureContribution(feature=clean_feature_name(name), contribution=round(float(value), 3))
                        )
                explainability_data = sorted(raw_contributions, key=lambda x: abs(x.contribution), reverse=True)
            except Exception as se:
                logger.error(f"Error executing SHAP computation: {se}", exc_info=True)

        return {
            "predicted_delay_minutes": round(predicted_delay_minutes, 2),
            "is_delayed": is_delayed,
            "explainability": explainability_data,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during prediction: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Prediction failed: {e}") from e


@app.post("/reload", status_code=status.HTTP_200_OK)
async def reload_assets():
    logger.info("Manual reload triggered for model and preprocessor.")
    load_model_and_preprocessor()
    if model is not None and preprocessor is not None:
        return {"status": "success", "message": "Model and preprocessor reloaded successfully."}
    else:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Reload attempted but one or more components failed to load.",
        )


@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    if model is not None and preprocessor is not None:
        return {"status": "ok", "message": "Model and preprocessor loaded successfully."}
    else:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model or preprocessor not loaded.")
