import logging
from contextlib import asynccontextmanager

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from mlflow import MlflowClient
from pydantic import BaseModel

from aeroscrape.config import get_app_config
from aeroscrape.data.clean import normalize_persian
from aeroscrape.features.build import FeatureSettings, add_calendar_features
from aeroscrape.models.delay_model import DelayModelBundle
from aeroscrape.serving.feature_lookup import FeatureLookup
from aeroscrape.storage.database import Database

logger = logging.getLogger(__name__)

model = None  # MLflow pyfunc: enforces the input signature
bundle: DelayModelBundle | None = None  # the same model unwrapped, for explanations and metadata
model_version: str | None = None
app_config = None
lookup: FeatureLookup | None = None

EXPLANATION_UNITS = "log-odds of a delay over 15 minutes, from the uncalibrated LightGBM classifier"


def load_model():
    """Load the registered delay model. Leaves the service unready if it can't be loaded."""
    global model, bundle, model_version, app_config

    try:
        app_config = get_app_config()
        mlflow.set_tracking_uri(app_config["mlflow"]["tracking_uri"])
        name = app_config["pipeline"]["model_registry_name"]
        alias = app_config["pipeline"]["model_alias_for_api"]
        model = mlflow.pyfunc.load_model(f"models:/{name}@{alias}")
        bundle = model.unwrap_python_model().bundle
        model_version = MlflowClient().get_model_version_by_alias(name, alias).version
        logger.info("Loaded %s version %s (%s variant)", name, model_version, bundle.variant)
    except Exception as e:
        logger.warning("Could not load the delay model: %s. Prediction will be unavailable.", e)
        model, bundle, model_version = None, None, None


def load_lookup():
    """Connect the feature lookup to the flight database (read-only role)."""
    global lookup
    try:
        lookup = FeatureLookup(Database(), FeatureSettings.from_config(get_app_config()))
    except Exception as e:
        logger.warning("Feature lookup unavailable: %s. Only request-time features will be used.", e)
        lookup = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    load_lookup()
    yield
    global model, bundle
    model, bundle = None, None


app = FastAPI(
    title="Flight Delay Prediction API",
    description="Calibrated probability of a delay over 15 minutes and a likely delay range.",
    version="2.0.0",
    lifespan=lifespan,
)


class FlightInput(BaseModel):
    airline: str
    airport: str
    destination_or_origin: str
    aircraft: str
    scheduled_datetime: str
    # With these, the lookup finds the scraped flight and the aircraft's rotation
    flight_number: str | None = None
    register: str | None = None


class DelayRange(BaseModel):
    p10: float
    p50: float
    p90: float


class FeatureContribution(BaseModel):
    feature: str
    value: str
    contribution: float


class PredictionOutput(BaseModel):
    model_version: str | None
    p_delay_over_15: float
    likely_delayed: bool
    decision_threshold: float
    delay_minutes: DelayRange
    top_factors: list[FeatureContribution]
    explanation_units: str
    warnings: list[str]
    # Kept for clients of the previous response format
    predicted_delay_minutes: float
    is_delayed: bool


def create_features(data: FlightInput, features: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """Build the model's input for one request with the training feature code.

    Returns the features and any warnings about information that couldn't be found.
    """
    try:
        scheduled = pd.to_datetime(data.scheduled_datetime, errors="raise")
    except Exception as e:
        logger.error(f"Error parsing scheduled_datetime '{data.scheduled_datetime}': {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid scheduled_datetime format. Use YYYY-MM-DD HH:MM:SS.",
        ) from e

    request = data.model_dump() | {"scheduled_datetime": str(scheduled)}
    warnings = []
    if lookup is not None:
        try:
            result = lookup.lookup(request)
            if data.flight_number and not result.matched_flight:
                warnings.append("Flight not found among scraped flights; features use the request only.")
            unknown = [c for c in features if pd.isna(result.features.iloc[0].get(c))]
            if unknown:
                warnings.append(f"Not known for this flight, filled by the model: {', '.join(unknown)}")
            return result.features.reindex(columns=features), warnings
        except Exception as e:
            logger.error("Feature lookup failed: %s", e, exc_info=True)

    # Without the lookup only the request's own fields are available
    df = pd.DataFrame([request])
    df["scheduled_datetime"] = scheduled
    for col in ("airline", "airport", "destination_or_origin", "aircraft"):
        df[col] = df[col].map(normalize_persian)
    df = add_calendar_features(df)
    missing = [col for col in features if col not in df.columns]
    warnings.append(f"Flight database unavailable; filled by the model: {', '.join(missing)}")
    return df.reindex(columns=features), warnings


@app.post("/predict", response_model=PredictionOutput, status_code=status.HTTP_200_OK)
async def predict(flight_data: FlightInput):
    if model is None or bundle is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model is not loaded.")

    try:
        features, warnings = create_features(flight_data, bundle.features.columns)
        X = bundle.prepare_input(features)
        prediction = model.predict(X).iloc[0]
        factors = bundle.explain(X, top=6)[0]
        p50 = round(float(prediction["p50"]), 1)
        likely = bool(prediction["likely_delayed"])
        return {
            "model_version": model_version,
            "p_delay_over_15": round(float(prediction["p_delay_over_15"]), 3),
            "likely_delayed": likely,
            "decision_threshold": bundle.threshold,
            "delay_minutes": {
                "p10": round(float(prediction["p10"]), 1),
                "p50": p50,
                "p90": round(float(prediction["p90"]), 1),
            },
            "top_factors": factors,
            "explanation_units": EXPLANATION_UNITS,
            "warnings": warnings,
            "predicted_delay_minutes": p50,
            "is_delayed": likely,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during prediction: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Prediction failed: {e}") from e


@app.post("/reload", status_code=status.HTTP_200_OK)
async def reload_assets():
    load_model()
    if model is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Reload failed; model not loaded.")
    return {"status": "success", "model_version": model_version}


@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    if model is None or bundle is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model is not loaded.")
    return {
        "status": "ok",
        "model_version": model_version,
        "variant": bundle.variant,
        "feature_lookup": "connected" if lookup is not None else "unavailable",
    }
