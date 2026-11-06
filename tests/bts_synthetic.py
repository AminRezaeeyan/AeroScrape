"""Synthetic flights in the exact layout of a BTS on-time zip, for tests that can't download BTS.

Aircraft fly rotations through ATL; a late arrival makes the next departure late, so the
delay-propagation features carry real signal, as they do in BTS data.
"""

import zipfile
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

OUTSTATIONS = ["BOS", "ORD", "DFW", "MIA", "DEN", "LGA"]
AIRLINES = ["DL", "WN", "AA"]


def _hhmm(minutes_after_midnight: int) -> int:
    """BTS clock format; midnight at the end of a day is written 2400."""
    m = minutes_after_midnight % (24 * 60)
    return 2400 if m == 0 and minutes_after_midnight > 0 else m // 60 * 100 + m % 60


def make_bts_frame(start: date, days: int, tails: int = 24, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for t in range(tails):
        tail = f"N{100 + t}DL"
        airline = AIRLINES[t % len(AIRLINES)]
        for d in range(days):
            day = start + timedelta(days=d)
            clock = 6 * 60 + 30 * (t % 6)  # first departure of the day, minutes after midnight
            carried = 0.0  # delay the aircraft carries from its previous leg
            for leg in range(4):
                origin, dest = ("ATL", OUTSTATIONS[(t + d + leg // 2) % len(OUTSTATIONS)])
                if leg % 2:
                    origin, dest = dest, origin
                block = 90 + 15 * (t % 3)
                dep_delay = max(-10.0, carried * 0.8 + rng.gamma(1.5, 8) - 6)
                arr_delay = dep_delay + rng.normal(0, 6)
                cancelled = rng.random() < 0.02
                diverted = not cancelled and rng.random() < 0.005
                rows.append(
                    {
                        "Year": day.year,
                        "Month": day.month,
                        "FlightDate": day.isoformat(),
                        "Reporting_Airline": airline,
                        "Tail_Number": tail if rng.random() > 0.01 else np.nan,
                        "Flight_Number_Reporting_Airline": 1000 + t * 10 + leg,
                        "Origin": origin,
                        "Dest": dest,
                        "CRSDepTime": _hhmm(clock),
                        "DepTime": np.nan if cancelled else _hhmm(int(clock + dep_delay)),
                        "DepDelay": np.nan if cancelled else round(dep_delay),
                        "CRSArrTime": _hhmm(clock + block),
                        "ArrTime": np.nan if cancelled or diverted else _hhmm(int(clock + block + arr_delay)),
                        "ArrDelay": np.nan if cancelled or diverted else round(arr_delay),
                        "Cancelled": float(cancelled),
                        "Diverted": float(diverted),
                        "Distance": 600.0,
                    }
                )
                carried = 0.0 if cancelled else max(0.0, arr_delay)
                clock += block + 45
    return pd.DataFrame(rows)


def write_bts_zip(path: Path, frame: pd.DataFrame) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "On_Time_Reporting_Carrier_On_Time_Performance_(1987_present)_2026_1.csv", frame.to_csv(index=False)
        )
        archive.writestr("readme.html", "<html>BTS readme</html>")
    return path
