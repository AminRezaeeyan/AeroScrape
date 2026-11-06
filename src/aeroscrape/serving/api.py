"""Flight delay prediction API, version 1.

POST /v1/predict          one flight
POST /v1/predict/batch    up to 500 flights, scored together
GET  /v1/model            the served model and the category values it knows
POST /v1/reload           check the registry for a newly promoted model now
GET  /health              the process is up
GET  /ready               the model and the feature lookup are loaded
GET  /metrics             Prometheus metrics

Prediction routes need the X-API-Key header when AEROSCRAPE_API_KEY is set. Every scored
flight is logged to the predictions table after the response is sent, for monitoring; the
X-Traffic-Source header (client, replay or loadtest) labels synthetic traffic.
"""

import asyncio
import logging
import os
import secrets
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Annotated, Any

import mlflow
import pandas as pd
from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException, Query, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from mlflow import MlflowClient
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, ConfigDict, Field

from aeroscrape.config import get_app_config
from aeroscrape.data.clean import normalize_persian
from aeroscrape.features.build import FeatureSettings, add_calendar_features
from aeroscrape.models.delay_model import DelayModelBundle
from aeroscrape.monitoring.prediction_log import TRAFFIC_SOURCES, PredictionLog, PredictionRecord, build_records
from aeroscrape.serving import metrics
from aeroscrape.serving.feature_lookup import FeatureLookup
from aeroscrape.serving.logs import configure_logging, request_id
from aeroscrape.storage.database import Database

logger = logging.getLogger(__name__)

EXPLANATION_UNITS = "log-odds of a delay over 15 minutes, from the uncalibrated LightGBM classifier"
MAX_BATCH = 500
CHECKED_CATEGORIES = ("airline", "airport", "destination_or_origin", "aircraft")


@dataclass(frozen=True)
class ServedModel:
    model: Any  # MLflow pyfunc: enforces the input signature
    bundle: DelayModelBundle  # the same model unwrapped, for explanations and metadata
    version: str
    known: dict[str, set[str]] = field(default_factory=dict)  # categories seen in training


# Replaced in a single assignment when a new version is promoted, so a request always uses
# one consistent model; each request reads it once at the start.
served: ServedModel | None = None
app_config = None
lookup: FeatureLookup | None = None
prediction_log: PredictionLog | None = None


def _serve(model: Any, version: str) -> ServedModel:
    bundle = model.unwrap_python_model().bundle
    metrics.set_model_info(version, bundle.variant)
    return ServedModel(model, bundle, version, bundle.known_categories())


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
        served = _serve(mlflow.pyfunc.load_model(f"models:/{name}/{version}"), version)
        logger.info("Model loaded", extra={"model": name, "model_version": version, "variant": served.bundle.variant})
    except Exception as e:
        logger.warning("Could not load the delay model; prediction is unavailable", extra={"error": str(e)})
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
        previous = served.version if served else None
        served = _serve(mlflow.pyfunc.load_model(f"models:/{name}/{version}"), version)
        metrics.MODEL_RELOADS.inc()
        logger.info("Swapped to a newly promoted model", extra={"from_version": previous, "model_version": version})
        return True
    except Exception as e:
        logger.warning(
            "Registry check failed; keeping the current model",
            extra={"model_version": served.version if served else None, "error": str(e)},
        )
        return False


async def poll_registry(interval_seconds: float) -> None:
    while True:
        await asyncio.sleep(interval_seconds)
        await asyncio.to_thread(refresh_if_promoted)


def load_lookup() -> None:
    """Connect the feature lookup and the prediction log to the flight database (API role)."""
    global lookup, prediction_log
    try:
        db = Database()
        lookup = FeatureLookup(db, FeatureSettings.from_config(get_app_config()))
        prediction_log = PredictionLog(db)
    except Exception as e:
        logger.warning("Feature lookup unavailable; only request fields will be used", extra={"error": str(e)})
        lookup = prediction_log = None


def write_prediction_log(records: list[PredictionRecord]) -> None:
    """Runs after the response is sent: a logging failure never fails or slows a prediction."""
    if prediction_log is None:
        return
    try:
        prediction_log.write(records)
    except Exception as e:
        metrics.LOG_FAILURES.inc()
        logger.warning("Could not log predictions", extra={"error": str(e), "flights": len(records)})


API_KEY = os.getenv("AEROSCRAPE_API_KEY")


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging(os.getenv("LOG_LEVEL", "INFO"))
    if not API_KEY:
        logger.warning("AEROSCRAPE_API_KEY is not set: prediction routes accept requests without a key")
    load_model()
    load_lookup()
    interval = get_app_config().get("serving", {}).get("registry_poll_seconds", 120)
    poller = asyncio.create_task(poll_registry(interval))
    yield
    poller.cancel()


app = FastAPI(
    title="Flight Delay Prediction API",
    description="Calibrated probability of a delay over 15 minutes and a likely delay range.",
    version="3.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in os.getenv("AEROSCRAPE_CORS_ORIGINS", "http://localhost:8501").split(",")],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-API-Key", "X-Request-ID", "X-Traffic-Source"],
)
Instrumentator(excluded_handlers=["/metrics", "/health", "/ready"]).instrument(app).expose(app, include_in_schema=False)


@app.middleware("http")
async def request_context(request: Request, call_next):
    """Give every request an ID (the caller's X-Request-ID if sent) and log it as one JSON line."""
    rid = request.headers.get("X-Request-ID") or uuid.uuid4().hex
    token = request_id.set(rid)
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        logger.exception("Unhandled error", extra={"path": request.url.path})
        response = JSONResponse({"detail": "Internal error", "request_id": rid}, status_code=500)
    latency_ms = round((time.perf_counter() - start) * 1000, 1)
    response.headers["X-Request-ID"] = rid
    if request.url.path not in ("/health", "/ready", "/metrics"):
        logger.info(
            "request",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "latency_ms": latency_ms,
                **getattr(request.state, "log", {}),
            },
        )
    request_id.reset(token)
    return response


def require_api_key(x_api_key: Annotated[str | None, Header()] = None) -> None:
    if API_KEY and not (x_api_key and secrets.compare_digest(x_api_key, API_KEY)):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing or invalid X-API-Key")


# ---------------------------------------------------------------------------- schemas


class FlightInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid", populate_by_name=True)

    airline: str = Field(min_length=1, max_length=100)
    airport: str = Field(min_length=1, max_length=100)
    destination_or_origin: str = Field(min_length=1, max_length=100)
    aircraft: str | None = Field(None, max_length=50)
    scheduled_datetime: datetime = Field(description="Local scheduled departure, e.g. 2026-11-03 08:05:00")
    # With these, the lookup finds the scraped flight and the aircraft's rotation
    flight_number: str | None = Field(None, max_length=20)
    registration: str | None = Field(None, alias="register", max_length=20)


class BatchInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    flights: list[FlightInput] = Field(min_length=1, max_length=MAX_BATCH)


class DelayRange(BaseModel):
    p10: float
    p50: float
    p90: float


class FeatureContribution(BaseModel):
    feature: str
    value: str
    contribution: float


class Prediction(BaseModel):
    p_delay_over_15: float
    likely_delayed: bool
    delay_minutes: DelayRange
    top_factors: list[FeatureContribution] | None = Field(None, description="Only with ?explain=true")
    warnings: list[str]


class PredictionResponse(Prediction):
    model_version: str
    decision_threshold: float
    explanation_units: str | None = None


class BatchResponse(BaseModel):
    model_version: str
    decision_threshold: float
    explanation_units: str | None = None
    predictions: list[Prediction]


# ---------------------------------------------------------------------------- prediction


def _request_dict(flight: FlightInput) -> dict[str, Any]:
    return flight.model_dump(by_alias=True) | {"scheduled_datetime": str(pd.Timestamp(flight.scheduled_datetime))}


def build_features(flights: list[FlightInput], columns: list[str]) -> tuple[pd.DataFrame, list[list[str]]]:
    """Model input for the flights, built with the training feature code, plus per-flight warnings."""
    requests = [_request_dict(f) for f in flights]
    if lookup is not None:
        try:
            with metrics.FEATURE_LOOKUP_SECONDS.time():
                result = lookup.lookup_many(requests)
            features = result.features.reindex(columns=columns)
            warnings = []
            for i, flight in enumerate(flights):
                notes = []
                if flight.flight_number and not result.matched[i]:
                    metrics.UNMATCHED_FLIGHTS.inc()
                    notes.append("Flight not found among scraped flights; features use the request only.")
                unknown = [c for c in columns if pd.isna(features.iloc[i][c])]
                if unknown:
                    notes.append(f"Not known for this flight, filled by the model: {', '.join(unknown)}")
                warnings.append(notes)
            return features, warnings
        except Exception as e:
            metrics.LOOKUP_FAILURES.inc()
            logger.error("Feature lookup failed; using request fields only", extra={"error": str(e)}, exc_info=True)

    df = pd.DataFrame(requests)
    df["scheduled_datetime"] = pd.to_datetime(df["scheduled_datetime"])
    for col in CHECKED_CATEGORIES:
        df[col] = df[col].map(normalize_persian)
    df = add_calendar_features(df)
    missing = [c for c in columns if c not in df.columns]
    note = f"Flight database unavailable; filled by the model: {', '.join(missing)}"
    return df.reindex(columns=columns), [[note] for _ in flights]


def _unseen(current: ServedModel, X: pd.DataFrame) -> list[list[str]]:
    """Category values the model never saw in training: it ignores them, so the caller should know."""
    checked = [c for c in CHECKED_CATEGORIES if c in current.known]
    return [
        [f"unseen {col}: {row[col]}" for col in checked if row[col] not in current.known[col] and row[col] != "nan"]
        for _, row in X[checked].iterrows()
    ]


def _traffic_source(request: Request) -> str:
    source = request.headers.get("X-Traffic-Source", "client").lower()
    return source if source in TRAFFIC_SOURCES else "client"


def _record_metrics(current: ServedModel, X: pd.DataFrame, out: pd.DataFrame, source: str, endpoint: str) -> None:
    known_airports = current.known.get("airport")
    for airport, delayed, p50, p in zip(
        X["airport"], out["likely_delayed"], out["p50"], out["p_delay_over_15"], strict=True
    ):
        label = airport if known_airports is None or airport in known_airports else "other"
        metrics.PREDICTIONS.labels(label, str(bool(delayed)).lower(), source, endpoint).inc()
        metrics.PREDICTED_DELAY.observe(float(p50))
        metrics.PREDICTED_PROBABILITY.observe(float(p))


def predict_flights(
    flights: list[FlightInput], explain: bool, request: Request, background: BackgroundTasks, endpoint: str
) -> tuple[ServedModel, list[Prediction]]:
    current = served
    if current is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model is not loaded.")
    bundle = current.bundle
    features, warnings = build_features(flights, bundle.features.columns)
    X = bundle.prepare_input(features)
    with metrics.MODEL_SECONDS.time():
        out = current.model.predict(X)  # one vectorized call for the whole batch
    factors: list[list[dict[str, Any]] | None] = [None] * len(flights)
    if explain:
        with metrics.SHAP_SECONDS.time():
            factors = list(bundle.explain(X, top=6))
    unseen = _unseen(current, X)
    for notes in unseen:
        for note in notes:
            metrics.UNSEEN_CATEGORIES.labels(note.split(":", 1)[0].removeprefix("unseen ")).inc()
    contributions = [None if rows is None else [FeatureContribution(**f) for f in rows] for rows in factors]
    source = _traffic_source(request)
    _record_metrics(current, X, out, source, endpoint)
    records = build_records(
        request_id.get() or uuid.uuid4().hex, source, [_request_dict(f) for f in flights], X, out, current.version
    )
    background.add_task(write_prediction_log, records)

    predictions = [
        Prediction(
            p_delay_over_15=round(float(row["p_delay_over_15"]), 3),
            likely_delayed=bool(row["likely_delayed"]),
            delay_minutes=DelayRange(
                p10=round(float(row["p10"]), 1), p50=round(float(row["p50"]), 1), p90=round(float(row["p90"]), 1)
            ),
            top_factors=contributions[i],
            warnings=unseen[i] + warnings[i],
        )
        for i, (_, row) in enumerate(out.iterrows())
    ]
    request.state.log = {
        "model_version": current.version,
        "flights": len(flights),
        "explain": explain,
        "mean_p_delay": round(float(out["p_delay_over_15"].mean()), 3),
        "mean_p50_minutes": round(float(out["p50"].mean()), 1),
    }
    return current, predictions


Explain = Annotated[bool, Query(description="Add per-flight SHAP contributions (slower)")]


@app.post("/v1/predict", response_model=PredictionResponse, dependencies=[Depends(require_api_key)])
def predict(flight: FlightInput, request: Request, background: BackgroundTasks, explain: Explain = False):
    current, [prediction] = predict_flights([flight], explain, request, background, "single")
    return PredictionResponse(
        **prediction.model_dump(),
        model_version=current.version,
        decision_threshold=current.bundle.threshold,
        explanation_units=EXPLANATION_UNITS if explain else None,
    )


@app.post("/v1/predict/batch", response_model=BatchResponse, dependencies=[Depends(require_api_key)])
def predict_batch(batch: BatchInput, request: Request, background: BackgroundTasks, explain: Explain = False):
    current, predictions = predict_flights(batch.flights, explain, request, background, "batch")
    return BatchResponse(
        model_version=current.version,
        decision_threshold=current.bundle.threshold,
        explanation_units=EXPLANATION_UNITS if explain else None,
        predictions=predictions,
    )


class ModelInfo(BaseModel):
    model_version: str
    variant: str
    decision_threshold: float
    trained_on: list[str] | None = None
    categories: dict[str, list[str]] = Field(description="Values of each category the model was trained on")


@app.get("/v1/model", response_model=ModelInfo, dependencies=[Depends(require_api_key)])
def model_info():
    """The served model and the category values it knows, e.g. for a client's dropdowns."""
    current = served
    if current is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model is not loaded.")
    trained_on = getattr(current.bundle, "metadata", {}).get("trained_on")
    return ModelInfo(
        model_version=current.version,
        variant=current.bundle.variant,
        decision_threshold=current.bundle.threshold,
        trained_on=[str(t) for t in trained_on] if trained_on else None,
        categories={col: sorted(v for v in values if v != "nan") for col, values in current.known.items()},
    )


@app.post("/v1/reload", dependencies=[Depends(require_api_key)])
async def reload_model():
    """Check the registry now instead of waiting for the next poll."""
    swapped = await asyncio.to_thread(refresh_if_promoted)
    if served is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Reload failed; model not loaded.")
    return {"status": "success", "model_version": served.version, "swapped": swapped}


@app.get("/health")
async def health():
    """Liveness: the process is up. Says nothing about the model."""
    return {"status": "alive"}


@app.get("/ready")
async def ready():
    """Readiness: the model and the feature lookup are loaded, so predictions are at full quality."""
    current = served
    body = {
        "model_version": current.version if current else None,
        "variant": current.bundle.variant if current else None,
        "feature_lookup": "connected" if lookup is not None else "unavailable",
        "auth": "api_key" if API_KEY else "disabled",
    }
    if current is None or lookup is None:
        return JSONResponse({"status": "not_ready", **body}, status_code=status.HTTP_503_SERVICE_UNAVAILABLE)
    return {"status": "ready", **body}
