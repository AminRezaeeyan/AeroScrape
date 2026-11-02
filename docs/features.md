# Features

All features are computed in [`features/build.py`](../src/aeroscrape/features/build.py) by `build_features(targets, context)`. `targets` are the departures to predict and `context` is every known flight, arrivals included. Groups and model variants are defined in `config.yaml` under `features`.

## When a feature is known

A prediction is only useful before the flight, so every feature is restricted to what was known at prediction time. There are two model variants:

| Variant | Prediction time | Groups |
| --- | --- | --- |
| `booking` | when the flight is booked | calendar, identity, rotation, rates |
| `pre_departure` | 1 hour before departure (`prediction_lead_minutes`) | the above + live propagation |

Congestion is computed and kept in the ablation, but left out of both variants. The counts come from what was scraped, not from a full schedule, and scraped departures per day varied from 277 to 688 depending on how often the scraper ran, while training rows stayed near 240 a day. The count correlates 0.70 with the day's delay share, so it measures scraping effort rather than traffic, and a model using it (the LightGBM classifier ranked it first) would not carry over to live scraping.

The API serves `booking` for now (`features.served_variant`), because it can't yet look up other flights at request time (Phase 4).

## Groups

| Group | Features | Known because |
| --- | --- | --- |
| calendar | hour, minute, weekday, Thursday/Friday weekend, time of day | from the schedule |
| identity | airline, destination, aircraft type, airport | from the schedule |
| rotation | `has_inbound`, `scheduled_turnaround_min` (minutes between the aircraft's scheduled arrival and this departure), `nth_flight_of_day_for_aircraft` | from the schedule: aircraft registration, inbound arrival within 6 h |
| congestion | `departures_same_hour` at the airport, `airline_departures_today` | from the schedule |
| rates | `airline_delay_rate_7d`, `route_median_delay_7d` | outcomes of the 7 days *before* the flight's date only, shrunk toward the overall rate for rare airlines and routes (`(n * own + 20 * overall) / (n + 20)`) |
| live propagation | `prev_departure_delay_min` (same aircraft's previous departure, any airport, within 12 h), `inbound_delay_min`, `inbound_overdue_min` | only if the earlier flight's actual time was before the prediction time. If the inbound hadn't landed by then, `inbound_overdue_min` records how overdue it already was, a lower bound on its delay |

`tests/test_features.py` checks these rules, including a leakage test: changing every outcome from a flight's own day onward must not change its rate features.

## Left out

Season and the Nowruz and holiday flags (`add_seasonal_features`) are not used. The data covers three weeks of late May and June 2025, so season is constant and there is no Nowruz; a model can't learn from a constant column. The code stays, so a group can be enabled once the data spans a year. The holiday dates are Gregorian approximations and lunar holidays will need a Hijri calendar.

## Coverage

Coverage is the main limit of the 2025 data:

- **Aircraft registrations are published only at Mehrabad** (55% of its departures, almost none elsewhere), so rotation and live features exist for about a third of training rows.
- **Mehrabad arrivals in the 2025 scrapes have no landing time**, so `inbound_delay_min` and `inbound_overdue_min` are empty. The current site does show landing times, so they will fill as new data is scraped. `prev_departure_delay_min` is the live signal that works today.

## What each group adds

From [results/evaluation.md](results/evaluation.md): each row adds one group to the row above, with default parameters, on the 687-flight test set. Changes are paired bootstrap differences against the first row.

| Features | MAE (min) | Change in MAE | Brier | PR-AUC |
| --- | --- | --- | --- | --- |
| calendar + identity | 23.8 | reference | 0.209 | 0.763 |
| + rotation | 23.8 | -0.0 [-0.3, +0.3] | 0.209 | 0.764 |
| + congestion | 24.1 | +0.3 [-0.2, +0.7] | 0.209 | 0.765 |
| + rates | 24.3 | +0.5 [-0.1, +1.0] | 0.211 | 0.764 |
| + live propagation | 23.2 | -0.6 [-1.4, +0.1] | 0.208 | 0.778 |

Reading it honestly:

- The schedule-based groups (rotation, congestion, rates) add nothing measurable on this data, and rates may slightly hurt. With three weeks of history the 7-day rates are noisy, and airline identity already carries most of what they say.
- The previous delay of the same aircraft is the strongest single feature (correlation 0.37 with the delay, against 0.17 or less for every other numeric feature), and the 1-hour-before model is the only one whose gain shows here, even with that feature known for a third of flights.
- Most of the improvement over Phase 1 comes from the recovered data: the Phase 1 feature set scores 23.8 MAE on the larger dataset, where the tuned Phase 1 model scored 24.4 on the old one (different test sets, so this is indicative only).
