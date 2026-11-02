import asyncio
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

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


@dataclass(frozen=True)
class ServedModel:
    model: Any  # MLflow pyfunc: enforces the input signature
    bundle: DelayModelBundle  # the same model unwrapped, for explanations and metadata
    version: str


# Replaced in a single assignment when a new version is promoted, so a request always uses
# one consistent model; each request reads it once at the start.
served: ServedModel | None = None
app_config = None
lookup: FeatureLookup | None = None

EXPLANATION_UNITS = "log-odds of a delay over 15 minutes, from the uncalibrated LightGBM classifier"


def _registry() -> tuple[MlflowClient, str, str]:
    global app_config
    app_config = get_app_config()
    mlflow.set_tracking_uri(app_config["mlflow"]["tracking_uri"])
    pipeline = app_config["pipeline"]
    return MlflowClient(), pipeline["model_registry_name"], pipeline["model_alias_for_api"]


def load_model() -> None:
    """Load the model the production alias points to. Leaves the service unready if it can't."""
    global served
    try:
        client, name, alias = _registry()
        version = str(client.get_model_version_by_alias(name, alias).version)
        model = mlflow.pyfunc.load_model(f"models:/{name}/{version}")
        served = ServedModel(model, model.unwrap_python_model().bundle, version)
        logger.info("Loaded %s version %s (%s variant)", name, version, served.bundle.variant)
    except Exception as e:
        logger.warning("Could not load the delay model: %s. Prediction will be unavailable.", e)
        served = None


def refresh_if_promoted() -> bool:
    """Load the production model if the alias moved to another version. Returns True if it swapped.

    The new model is fully loaded before the swap; on any error the current model keeps serving.
    """
    global served
    try:
        client, name, alias = _registry()
        version = str(client.get_model_version_by_alias(name, alias).version)
        if served is not None and served.version == version:
            return False
        model = mlflow.pyfunc.load_model(f"models:/{name}/{version}")
        previous = served.version if served else None
        served = ServedModel(model, model.unwrap_python_model().bundle, version)
        logger.info("Swapped %s from version %s to %s", name, previous, version)
        return True
    except Exception as e:
        logger.warning("Registry check failed, keeping version %s: %s", served.version if served else None, e)
        return False


async def poll_registry(interval_seconds: float) -> None:
    while True:
        await asyncio.sleep(interval_seconds)
        await asyncio.to_thread(refresh_if_promoted)


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
    interval = get_app_config().get("serving", {}).get("registry_poll_seconds", 120)
    poller = asyncio.create_task(poll_registry(interval))
    yield
    poller.cancel()


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
    current = served
    if current is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model is not loaded.")
    bundle = current.bundle

    try:
        features, warnings = create_features(flight_data, bundle.features.columns)
        X = bundle.prepare_input(features)
        prediction = current.model.predict(X).iloc[0]
        factors = bundle.explain(X, top=6)[0]
        p50 = round(float(prediction["p50"]), 1)
        likely = bool(prediction["likely_delayed"])
        return {
            "model_version": current.version,
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
    """Check the registry now instead of waiting for the next poll."""
    swapped = await asyncio.to_thread(refresh_if_promoted)
    if served is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Reload failed; model not loaded.")
    return {"status": "success", "model_version": served.version, "swapped": swapped}


@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    current = served
    if current is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model is not loaded.")
    return {
        "status": "ok",
        "model_version": current.version,
        "variant": current.bundle.variant,
        "feature_lookup": "connected" if lookup is not None else "unavailable",
    }
