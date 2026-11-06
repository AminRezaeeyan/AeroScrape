"""Print one markdown table from the CSV files of several headless Locust runs.

    python tools/locust_report.py 8 reports/locust/single reports/locust/explain reports/locust/batch

The first argument is the number of users, for the heading.

Each argument is a --csv prefix; the run's own `<prefix>_stats.csv` is read.
"""

import csv
import sys


def main(prefixes: list[str], users: str) -> None:
    print(f"Concurrency {users} users\n")
    print("| Scenario | Requests | p50 (ms) | p95 (ms) | p99 (ms) | Requests/s | Flights/s | Failures |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for prefix in prefixes:
        with open(f"{prefix}_stats.csv", newline="", encoding="utf-8") as f:
            row = next(r for r in csv.DictReader(f) if r["Name"] != "Aggregated")
        rps = float(row["Requests/s"])
        per_request = int(row["Name"].rsplit(" ", 1)[1]) if row["Name"].startswith("batch of") else 1
        print(
            f"| {row['Name']} | {row['Request Count']} | {row['50%']} | {row['95%']} | {row['99%']} | "
            f"{rps:.1f} | {rps * per_request:.0f} | {row['Failure Count']} |"
        )


if __name__ == "__main__":
    main(sys.argv[2:], sys.argv[1])
