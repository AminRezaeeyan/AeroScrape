"""Prometheus metrics of the prediction API, exposed at /metrics.

Request count and latency per route come from prometheus-fastapi-instrumentator; these are
the model-specific ones. Labels stay low-cardinality: an airport label only takes airports
the model was trained on, everything else is "other", so callers can't create series at will.
"""

from prometheus_client import Counter, Gauge, Histogram

PREDICTIONS = Counter(
    "predictions_total", "Flights scored", ["airport", "likely_delayed", "traffic_source", "endpoint"]
)
PREDICTED_DELAY = Histogram(
    "predicted_delay_minutes",
    "Predicted median delay of each scored flight",
    buckets=[0, 5, 10, 15, 30, 60, 120, 240],
)
PREDICTED_PROBABILITY = Histogram(
    "predicted_delay_probability",
    "Calibrated probability of a delay over 15 minutes",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)
FEATURE_LOOKUP_SECONDS = Histogram("feature_lookup_seconds", "Time to build the features of a request")
MODEL_SECONDS = Histogram("model_predict_seconds", "Time in the model's predict call per request")
SHAP_SECONDS = Histogram("shap_seconds", "Time to compute SHAP contributions per request")
UNSEEN_CATEGORIES = Counter("unseen_categories_total", "Request values the model never saw in training", ["column"])
UNMATCHED_FLIGHTS = Counter("unmatched_flights_total", "Requested flight numbers not found among scraped flights")
LOOKUP_FAILURES = Counter("feature_lookup_failures_total", "Requests scored without the flight database")
LOG_FAILURES = Counter("prediction_log_failures_total", "Batches of predictions that could not be logged")
MODEL_INFO = Gauge("model_version_info", "The served model version (value 1)", ["version", "variant"])
MODEL_RELOADS = Counter("model_reloads_total", "Swaps to a newly promoted model")


def set_model_info(version: str, variant: str) -> None:
    MODEL_INFO.clear()
    MODEL_INFO.labels(version=version, variant=variant).set(1)
