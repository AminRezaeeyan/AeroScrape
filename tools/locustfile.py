"""Locust load test for the prediction API.

One user class per scenario, so each can be measured on its own:

    locust -f tools/locustfile.py SingleFlight --headless -u 8 -r 8 -t 30s --host http://localhost:8000

`make loadtest` runs all three and prints one table. Flights are tomorrow's, so the feature
lookup does its full work against the flight database. AEROSCRAPE_API_KEY is sent as X-API-Key.
"""

import os
import random
from datetime import date, timedelta

from locust import HttpUser, constant, task

AIRLINES = ["ماهان", "ایران ایر", "آتا", "کاسپین", "قشم ایر", "وارش"]
ROUTES = [
    ("فرودگاه مهرآباد", "مشهد"),
    ("فرودگاه مهرآباد", "کیش"),
    ("فرودگاه مهرآباد", "شیراز"),
    ("فرودگاه مشهد", "تهران"),
    ("فرودگاه شیراز", "تهران"),
]
BATCH_SIZE = int(os.getenv("LOADTEST_BATCH_SIZE", "50"))
HEADERS = {"X-API-Key": os.environ["AEROSCRAPE_API_KEY"]} if os.getenv("AEROSCRAPE_API_KEY") else {}


def flight() -> dict:
    airport, destination = random.choice(ROUTES)
    day = date.today() + timedelta(days=1)
    return {
        "airline": random.choice(AIRLINES),
        "airport": airport,
        "destination_or_origin": destination,
        "aircraft": random.choice(["737", "A320", "MD83", "F100"]),
        "scheduled_datetime": f"{day} {random.randint(5, 23):02d}:{random.choice([0, 15, 30, 45]):02d}:00",
    }


class SingleFlight(HttpUser):
    """One flight per request, no explanation: the common case."""

    wait_time = constant(0)

    @task
    def predict(self) -> None:
        self.client.post("/v1/predict", json=flight(), headers=HEADERS, name="single flight")


class ExplainedFlight(HttpUser):
    """One flight with SHAP contributions."""

    wait_time = constant(0)

    @task
    def predict(self) -> None:
        self.client.post("/v1/predict?explain=true", json=flight(), headers=HEADERS, name="single flight, explain=true")


class BatchOfFlights(HttpUser):
    """A batch of flights scored in one request."""

    wait_time = constant(0)

    @task
    def predict(self) -> None:
        body = {"flights": [flight() for _ in range(BATCH_SIZE)]}
        self.client.post("/v1/predict/batch", json=body, headers=HEADERS, name=f"batch of {BATCH_SIZE}")
