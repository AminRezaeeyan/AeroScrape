"""Replay scraped departures through the API as if they were live traffic.

    python tools/replay.py --days 7 --api-key ...          # last week's departures
    python tools/replay.py --days 7 --night --api-key ...  # the same, moved to 00:00-04:59

The predictions are logged like any other (labelled traffic_source=replay), so the monitor
DAG can match them to the delays that already happened and check drift without waiting
for real traffic. --night drops flight numbers and moves every flight to night hours: an
input shift the drift check should catch. Run on the host; it reads the flight database
through the published port (DB_HOST_PORT in .env).
"""

import argparse
import os
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

import httpx
import pandas as pd
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]


def departures(days: int) -> pd.DataFrame:
    env = dotenv_values(ROOT / ".env")
    os.environ["DB_HOST"] = "localhost"
    os.environ["DB_PORT"] = str(env.get("DB_HOST_PORT") or 5433)
    from aeroscrape.data.clean import clean_flights
    from aeroscrape.storage.database import Database

    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    raw = Database().fetch_flights(today - timedelta(days=days), today + timedelta(days=1))
    if raw.empty:
        return raw
    flights = clean_flights(raw)
    return flights[flights["flight_type"] == "Domestic Departure"].reset_index(drop=True)


def requests_for(flights: pd.DataFrame, night: bool, seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    out = []
    for f in flights.itertuples(index=False):
        scheduled = pd.Timestamp(f.scheduled_datetime)
        body = {
            "airline": f.airline,
            "airport": f.airport,
            "destination_or_origin": f.destination_or_origin,
            "aircraft": f.aircraft if isinstance(f.aircraft, str) else None,
            "scheduled_datetime": str(scheduled),
        }
        if night:
            body["scheduled_datetime"] = str(scheduled.replace(hour=rng.randint(0, 4)))
        else:
            body["flight_number"] = f.flight_number
            if isinstance(f.register, str):
                body["register"] = f.register
        out.append(body)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--night", action="store_true", help="move every flight to night hours (a drift test)")
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--api-key", default=os.getenv("AEROSCRAPE_API_KEY"))
    parser.add_argument("--batch-size", type=int, default=100)
    args = parser.parse_args()

    flights = departures(args.days)
    if flights.empty:
        print(f"No departures scheduled in the last {args.days} days in the flight database.")
        return 1
    bodies = requests_for(flights, args.night)
    headers = {"X-Traffic-Source": "replay"} | ({"X-API-Key": args.api_key} if args.api_key else {})
    scored = 0
    with httpx.Client(base_url=args.url, headers=headers, timeout=120) as client:
        for start in range(0, len(bodies), args.batch_size):
            response = client.post("/v1/predict/batch", json={"flights": bodies[start : start + args.batch_size]})
            response.raise_for_status()
            scored += len(response.json()["predictions"])
            print(f"\r{scored}/{len(bodies)} flights scored", end="", flush=True)
    days = flights["scheduled_datetime"].dt.date
    print(
        f"\nReplayed {scored} departures scheduled {days.min()} to {days.max()}{' at night hours' if args.night else ''}."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
