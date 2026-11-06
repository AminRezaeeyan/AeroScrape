# Model card: flight-delay-model

## What it predicts

For one Iranian **domestic departure**: the probability that it leaves more than 15 minutes after its
scheduled time (calibrated), a likely delay in minutes (the median) with a 10–90% range, and a
"likely delayed" flag. Returned by `POST /v1/predict` with the model version and, on request, the
factors behind the probability.

Two variants are trained; the API serves the one in `features.served_variant`:

| Variant | Uses | Served |
| --- | --- | --- |
| At booking | Calendar, airline, airports, aircraft type, the aircraft's planned rotation, airline and route delays of the last 7 days | yes |
| 1 h before departure | The above, plus the actual delays of the same aircraft's earlier flights that day | no: needs registrations and live statuses, which most requests don't have |

## Intended use

- Planning around a domestic flight in Iran: when to leave, whether a connection is at risk.
- Demonstrating an end-to-end, monitored ML system.

**Not for:** operational decisions by airlines or airports, international flights, arrivals, or
compensation claims. It gives odds, not a schedule.

## Model

LightGBM. Three quantile regressors (10th, 50th and 90th percentile) for the minutes, with a
conformal margin chosen on out-of-fold predictions so the range holds about 80% of flights. A
LightGBM classifier for the probability, Platt-scaled on out-of-fold predictions from rolling-origin
cross-validation. The flag threshold is the lowest probability where out-of-fold precision reaches
75% (68% for the served version). One MLflow pyfunc holds all of it ([modeling.md](modeling.md)).

## Training data

Flights scraped from fids.airport.ir between 22 May and 12 June 2025: 3,574 domestic departures that
departed with a plausible delay (−60 to 720 minutes), from 26 airports, 59% of them at Mehrabad.
58.7% were delayed more than 15 minutes; the median delay was 20 minutes. Registrations, needed
for rotation features, exist for 59% of departures, almost all at Mehrabad; the aircraft's previous
delay, the strongest feature, is known for about a third.

The served version, 5, was trained on 22 May to 11 June 2025 (registry tags `training_from`,
`training_to`, `training_data_sha256`, `git_commit`).

## Evaluation

On 10 to 12 June 2025 (687 departures), with model choices made on the earlier weeks only.
Intervals are paired 95% bootstrap intervals against the best baseline.

| Metric | At booking | 1 h before | Best baseline |
| --- | --- | --- | --- |
| MAE of the median (min) | 24.0 | 23.0 | 24.7 |
| Brier score | 0.207 | 0.201 | 0.221 |
| Share inside the 10–90% range | 82.7% | 85.3% | 76.4% |
| Flag precision (target 75%) | 80–85% | | |

By segment (MAE at booking, minutes; [full table](results/evaluation.md#error-analysis)):

| Segment | Flights | MAE |
| --- | --- | --- |
| Mehrabad | 413 | 24.7 |
| Other airports | 245 | 23.3 |
| Inbound flight linked | 275 | 21.3 |
| No inbound flight linked | 412 | 25.8 |
| Airline آتا (79% delayed) | 104 | 31.3 |
| Airline ماهان (33% delayed) | 69 | 11.0 |

Version 5's own held-out window, scored when it was trained: MAE 20.2, Brier 0.225, coverage
86.7%. These are the "offline" numbers live monitoring compares against.

**Live, November 2026** (flights replayed through the API, matched to what happened): MAE 21.9 and
25.3 minutes on the two days with outcomes, Brier 0.24, coverage 84% and 93%.

## Known weaknesses

- **Three weeks of data from one season.** No holidays, no winter weather, no Nowruz.
- **Rotation features mostly missing.** Without a registration the model can't see the aircraft's
  earlier flights, and it is weakest exactly there (MAE 25.8 vs. 21.3 with an inbound flight).
- **Long delays are not predictable from these inputs.** Errors are largest for the airlines with
  the most delays. The range's upper end is wide (often over 100 minutes) for that reason.
- **The flag is conservative by design** (75% precision), so it misses many delayed flights.
- **Drift by November 2026:** 35% of recent flights are on airlines outside the training's 20 most
  common, the 7-day rates are missing for 38%, and registrations for 93%.

## When not to trust it

- The response lists **unseen categories** in `warnings` (an airline, airport, destination or
  aircraft type absent from training): the model ignores that field.
- `warnings` says the **flight database is unavailable**: only calendar and identity features were
  used.
- The flight is **not domestic**, or is more than about a week ahead (rates and rotation are
  unknown).
- Grafana shows **degraded live MAE, low coverage or input drift** for the served version.

## Maintenance

`monitor_model` checks live accuracy and drift daily; three degraded days or drift trigger
retraining, and the [promotion gate](operations.md#the-promotion-gate) decides. Every version
carries its git commit, training window and data hash in the registry.
