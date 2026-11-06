"""Load test the prediction API and report latency percentiles and throughput.

    python tools/loadtest.py --url http://localhost:8000 --api-key "$AEROSCRAPE_API_KEY"

Three scenarios, each run at a fixed concurrency for a fixed time: one flight per request,
one flight with SHAP explanations, and batches of 50 flights. Flights are tomorrow's, so the
feature lookup does its full work against the flight database.
"""

import argparse
import asyncio
import random
import statistics
import time
from dataclasses import dataclass, field
from datetime import date, timedelta

import httpx

AIRLINES = ["ماهان", "ایران ایر", "آتا", "کاسپین", "قشم ایر", "وارش"]
ROUTES = [
    ("فرودگاه مهرآباد", "مشهد"),
    ("فرودگاه مهرآباد", "کیش"),
    ("فرودگاه مهرآباد", "شیراز"),
    ("فرودگاه مشهد", "تهران"),
    ("فرودگاه شیراز", "تهران"),
]


def flight(rng: random.Random, day: date) -> dict:
    airport, destination = rng.choice(ROUTES)
    return {
        "airline": rng.choice(AIRLINES),
        "airport": airport,
        "destination_or_origin": destination,
        "aircraft": rng.choice(["737", "A320", "MD83", "F100"]),
        "scheduled_datetime": f"{day} {rng.randint(5, 23):02d}:{rng.choice([0, 15, 30, 45]):02d}:00",
    }


@dataclass
class Result:
    name: str
    latencies_ms: list[float] = field(default_factory=list)
    errors: int = 0
    elapsed_s: float = 0.0
    flights_per_request: int = 1

    def row(self) -> str:
        lat = sorted(self.latencies_ms)
        if not lat:
            return f"| {self.name} | 0 | - | - | - | - | - | {self.errors} |"
        q = statistics.quantiles(lat, n=100)
        rps = len(lat) / self.elapsed_s
        return (
            f"| {self.name} | {len(lat)} | {q[49]:.0f} | {q[94]:.0f} | {q[98]:.0f} | {rps:.1f} | "
            f"{rps * self.flights_per_request:.0f} | {self.errors} |"
        )


async def run(
    name: str, client: httpx.AsyncClient, path: str, body, seconds: float, concurrency: int, per: int
) -> Result:
    result = Result(name, flights_per_request=per)
    deadline = time.perf_counter() + seconds

    async def worker():
        while time.perf_counter() < deadline:
            start = time.perf_counter()
            try:
                response = await client.post(path, json=body())
                if response.status_code == 200:
                    result.latencies_ms.append((time.perf_counter() - start) * 1000)
                else:
                    result.errors += 1
            except httpx.HTTPError:
                result.errors += 1

    start = time.perf_counter()
    await asyncio.gather(*(worker() for _ in range(concurrency)))
    result.elapsed_s = time.perf_counter() - start
    return result


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--seconds", type=float, default=30)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=50)
    args = parser.parse_args()

    rng = random.Random(0)
    day = date.today() + timedelta(days=1)
    headers = {"X-API-Key": args.api_key} if args.api_key else {}
    async with httpx.AsyncClient(base_url=args.url, headers=headers, timeout=60) as client:
        await client.post("/v1/predict", json=flight(rng, day))  # warm the history cache
        results = [
            await run(
                "single flight", client, "/v1/predict", lambda: flight(rng, day), args.seconds, args.concurrency, 1
            ),
            await run(
                "single flight, explain=true",
                client,
                "/v1/predict?explain=true",
                lambda: flight(rng, day),
                args.seconds,
                args.concurrency,
                1,
            ),
            await run(
                f"batch of {args.batch_size}",
                client,
                "/v1/predict/batch",
                lambda: {"flights": [flight(rng, day) for _ in range(args.batch_size)]},
                args.seconds,
                args.concurrency,
                args.batch_size,
            ),
        ]
    print(f"Concurrency {args.concurrency}, {args.seconds:.0f} s per scenario, against {args.url}\n")
    print("| Scenario | Requests | p50 (ms) | p95 (ms) | p99 (ms) | Requests/s | Flights/s | Errors |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for r in results:
        print(r.row())


if __name__ == "__main__":
    asyncio.run(main())
