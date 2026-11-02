<!-- Snapshot of reports/evaluation.md from `make evaluate` on data/raw/data.csv (sha256 7b9ba655...). Regenerate after changing data, features or models. -->

# Evaluation report

Test window: 2025-06-10 to 2025-06-12 (687 flights). Models and baselines are fit on train+val and scored once on test. Brackets are 95% bootstrap percentile intervals over 1,000 resamples of the test set; differences are paired, using the same resamples for both models.

## Data

| Split | Rows | From | To |
| --- | --- | --- | --- |
| train | 2398 | 2025-05-22 00:20 | 2025-06-04 23:30 |
| val | 489 | 2025-06-06 04:30 | 2025-06-09 23:30 |
| test | 687 | 2025-06-10 04:40 | 2025-06-12 23:50 |

## Delay in minutes (regression)

Test delays: median 20 min, mean 34 min, max 537 min.

| Model | MAE (min) | MedAE (min) | Within 15 min |
| --- | --- | --- | --- |
| Global median (baseline) | 25.2 [21.7, 28.8] | 12.0 [11.0, 14.0] | 60.6% [56.8%, 64.8%] |
| Airline × time-of-day median (baseline) | 24.7 [21.4, 28.2] | 11.5 [11.0, 13.0] | 61.0% [57.2%, 64.6%] |
| **LightGBM (at booking)** | 23.8 [20.4, 27.4] | 11.6 [10.6, 12.6] | 60.7% [57.2%, 64.2%] |
| **LightGBM (1 h before departure)** | 22.8 [19.6, 26.3] | 10.8 [10.2, 12.2] | 61.7% [58.1%, 65.4%] |

**LightGBM (at booking) minus best baseline (Airline × time-of-day median), MAE (min):** -0.9 [-1.4, -0.4]. The interval excludes 0, so the model is better.

**LightGBM (1 h before departure) minus best baseline (Airline × time-of-day median), MAE (min):** -1.9 [-2.8, -1.2]. The interval excludes 0, so the model is better.

## Delayed more than 15 minutes (classification)

60.3% of test flights are delayed, so always predicting 'delayed' scores 0.603 accuracy. PR-AUC of a random ranking equals that rate.

| Model | Accuracy | PR-AUC | Brier |
| --- | --- | --- | --- |
| Majority class (baseline) | 0.603 [0.565, 0.640] | 0.603 [0.565, 0.640] | 0.240 [0.234, 0.246] |
| Airline delay rate (baseline) | 0.646 [0.608, 0.681] | 0.717 [0.670, 0.762] | 0.221 [0.210, 0.231] |
| **Logistic regression (at booking)** | 0.699 [0.664, 0.734] | 0.763 [0.717, 0.807] | 0.207 [0.195, 0.220] |
| **Logistic regression (1 h before departure)** | 0.699 [0.662, 0.734] | 0.777 [0.734, 0.817] | 0.205 [0.193, 0.217] |

**Logistic regression (at booking) minus best baseline (Airline delay rate), Brier:** -0.013 [-0.023, -0.004]. The interval excludes 0, so the model is better.

**Logistic regression (1 h before departure) minus best baseline (Airline delay rate), Brier:** -0.016 [-0.025, -0.006]. The interval excludes 0, so the model is better.

**Logistic regression (at booking) minus best baseline (Airline delay rate), PR-AUC:** 0.046 [0.010, 0.083]. The interval excludes 0, so the model is better.

**Logistic regression (1 h before departure) minus best baseline (Airline delay rate), PR-AUC:** 0.060 [0.027, 0.094]. The interval excludes 0, so the model is better.

## Feature ablation

Each row adds one feature group to the row above (LightGBM for minutes, logistic regression for the classifier, default parameters). Changes are paired differences against the first row; negative is better for MAE and Brier.

| Features | MAE (min) | Change in MAE | Brier | Change in Brier | PR-AUC |
| --- | --- | --- | --- | --- | --- |
| calendar + identity | 23.8 [20.6, 27.2] | reference | 0.209 [0.196, 0.224] | reference | 0.763 [0.716, 0.810] |
| + rotation | 23.8 [20.6, 27.2] | -0.0 [-0.3, +0.3] | 0.209 [0.196, 0.224] | +0.000 [-0.003, +0.002] | 0.764 [0.715, 0.810] |
| + congestion | 24.1 [20.9, 27.6] | +0.3 [-0.2, +0.7] | 0.209 [0.195, 0.225] | +0.000 [-0.004, +0.004] | 0.765 [0.717, 0.810] |
| + rates | 24.3 [21.1, 27.8] | +0.5 [-0.1, +1.0] | 0.211 [0.197, 0.226] | +0.001 [-0.003, +0.006] | 0.764 [0.717, 0.809] |
| + live_propagation | 23.2 [20.0, 26.7] | -0.6 [-1.4, +0.1] | 0.208 [0.194, 0.223] | -0.002 [-0.006, +0.003] | 0.778 [0.733, 0.819] |

## Feature coverage

Aircraft registrations are published only for Mehrabad departures, so rotation and live features exist mostly there. Inbound landing times are missing from the 2025 scrapes, so the inbound delay features are empty until newly scraped data fills them. Rates are empty for the first days of data.

| Group | Feature | Known in train+val | Known in test |
| --- | --- | --- | --- |
| rotation | `has_inbound` | 100% | 100% |
| rotation | `scheduled_turnaround_min` | 25% | 40% |
| rotation | `nth_flight_of_day_for_aircraft` | 59% | 60% |
| live_propagation | `prev_departure_delay_min` | 30% | 33% |
| live_propagation | `inbound_delay_min` | 0% | 0% |
| live_propagation | `inbound_overdue_min` | 0% | 0% |
| rates | `airline_delay_rate_7d` | 92% | 100% |
| rates | `route_median_delay_7d` | 92% | 100% |

## Tuning

Parameters were chosen by rolling-origin cross-validation on train+val (test unused):

- at booking: CV MAE 23.8 min, CV Brier 0.220
- 1 h before departure: CV MAE 23.5 min, CV Brier 0.220

## Data cleaning

Rows remaining after each cleaning step:

| Step | Rows |
| --- | --- |
| Raw scraped rows | 15,732 |
| With a scheduled time and flight number | 15,732 |
| After removing duplicate flights | 15,732 |
| Domestic departures | 8,091 |
| ... with status departed | 3,575 |
| ... with a departure time | 3,575 |
| ... with a delay in [-60, 720] min | 3,574 |
