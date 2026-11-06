<!-- Snapshot of reports/bts/evaluation.md from `make evaluate SOURCE=bts` at commit f253f48f on BTS Jan-May 2026 (ATL). Undefined metrics printed as "nan [nan, nan]" by that commit are shown as n/a, as later commits print them. -->

# Evaluation report

Test window: 2026-05-25 to 2026-05-31 (6241 flights). Models and baselines are fit on train+val and scored once on test. Brackets are 95% bootstrap percentile intervals over 500 resamples of the test set; differences are paired, using the same resamples for both models.

## Data

| Split | Rows | From | To |
| --- | --- | --- | --- |
| train | 112669 | 2026-01-01 05:10 | 2026-05-17 23:23 |
| val | 6021 | 2026-05-18 05:05 | 2026-05-24 23:20 |
| test | 6241 | 2026-05-25 05:00 | 2026-05-31 23:23 |

## Delay in minutes (regression)

Test delays: median -1 min, mean 15 min, max 715 min.

| Model | MAE (min) | MedAE (min) | Within 15 min |
| --- | --- | --- | --- |
| Global median (baseline) | 19.7 [18.8, 20.7] | 5.0 [5.0, 5.0] | 72.3% [71.4%, 73.4%] |
| Airline × time-of-day median (baseline) | 19.4 [18.5, 20.3] | 5.0 [5.0, 5.0] | 72.6% [71.6%, 73.7%] |
| **LightGBM median (at booking)** | 18.4 [17.5, 19.2] | 5.3 [5.1, 5.5] | 73.5% [72.5%, 74.5%] |
| **LightGBM median (1 h before departure)** | 14.3 [13.6, 15.1] | 4.4 [4.3, 4.6] | 78.4% [77.5%, 79.4%] |

**LightGBM median (at booking) minus best baseline (Airline × time-of-day median), MAE (min):** -1.0 [-1.2, -0.9]. The interval excludes 0, so the model is better.

**LightGBM median (1 h before departure) minus best baseline (Airline × time-of-day median), MAE (min):** -5.1 [-5.5, -4.7]. The interval excludes 0, so the model is better.

## Delay range: 10th to 90th percentile

A calibrated 10-90% interval contains about 80% of actual delays. Pinball loss scores all three percentiles at once and rewards narrow intervals that still cover.

| Model | Coverage of 10-90% interval | Mean width (min) | Pinball loss (min) |
| --- | --- | --- | --- |
| Global percentiles (baseline) | 80.8% [79.9%, 81.8%] | 47.0 [47.0, 47.0] | 7.33 [6.97, 7.73] |
| Airline × time-of-day percentiles (baseline) | 81.9% [81.0%, 83.0%] | 45.6 [45.1, 46.0] | 7.00 [6.65, 7.39] |
| **LightGBM percentiles (at booking)** | 75.4% [74.3%, 76.4%] | 45.5 [44.9, 46.0] | 6.78 [6.45, 7.14] |
| **LightGBM percentiles + conformal (at booking)** | 79.0% [78.0%, 80.0%] | 46.8 [46.2, 47.4] | 6.78 [6.44, 7.13] |
| **LightGBM percentiles (1 h before departure)** | 75.6% [74.6%, 76.7%] | 35.1 [34.4, 35.8] | 5.21 [4.92, 5.57] |
| **LightGBM percentiles + conformal (1 h before departure)** | 80.6% [79.7%, 81.6%] | 37.0 [36.2, 37.7] | 5.21 [4.92, 5.57] |

**LightGBM percentiles + conformal (at booking) minus best baseline (Airline × time-of-day percentiles), Pinball loss (min):** -0.22 [-0.31, -0.15]. The interval excludes 0, so the model is better.

**LightGBM percentiles + conformal (1 h before departure) minus best baseline (Airline × time-of-day percentiles), Pinball loss (min):** -1.79 [-1.97, -1.63]. The interval excludes 0, so the model is better.

## Delayed more than 15 minutes (classification)

26.0% of test flights are delayed, so always predicting 'delayed' scores 0.740 accuracy. PR-AUC of a random ranking equals that rate.
Accuracy, precision and recall here use a 0.5 cutoff for every model; the chosen decision threshold is evaluated in its own section below.

| Model | Accuracy | Precision | Recall | PR-AUC | Brier | ECE |
| --- | --- | --- | --- | --- | --- | --- |
| Majority class (baseline) | 0.740 [0.730, 0.751] | n/a | 0.000 [0.000, 0.000] | 0.260 [0.249, 0.270] | 0.197 [0.190, 0.203] | 0.063 [0.053, 0.073] |
| Airline delay rate (baseline) | 0.740 [0.730, 0.751] | n/a | 0.000 [0.000, 0.000] | 0.296 [0.281, 0.312] | 0.194 [0.188, 0.201] | 0.065 [0.055, 0.076] |
| **Logistic regression (at booking)** | 0.743 [0.733, 0.754] | 0.685 [0.565, 0.807] | 0.023 [0.016, 0.031] | 0.438 [0.415, 0.462] | 0.178 [0.172, 0.183] | 0.049 [0.039, 0.060] |
| **LightGBM, uncalibrated (at booking)** | 0.764 [0.754, 0.774] | 0.736 [0.684, 0.788] | 0.144 [0.129, 0.162] | 0.501 [0.477, 0.529] | 0.170 [0.164, 0.175] | 0.028 [0.021, 0.038] |
| **LightGBM, calibrated (at booking)** | 0.758 [0.748, 0.768] | 0.772 [0.714, 0.830] | 0.098 [0.083, 0.113] | 0.501 [0.477, 0.529] | 0.171 [0.165, 0.176] | 0.037 [0.030, 0.048] |
| **Logistic regression (1 h before departure)** | 0.818 [0.808, 0.828] | 0.747 [0.721, 0.775] | 0.458 [0.430, 0.482] | 0.665 [0.641, 0.688] | 0.136 [0.131, 0.142] | 0.024 [0.020, 0.034] |
| **LightGBM, uncalibrated (1 h before departure)** | 0.832 [0.824, 0.841] | 0.819 [0.795, 0.845] | 0.457 [0.432, 0.480] | 0.739 [0.721, 0.759] | 0.123 [0.117, 0.128] | 0.030 [0.024, 0.039] |
| **LightGBM, calibrated (1 h before departure)** | 0.833 [0.825, 0.842] | 0.815 [0.791, 0.842] | 0.465 [0.440, 0.488] | 0.739 [0.721, 0.759] | 0.123 [0.117, 0.128] | 0.028 [0.022, 0.037] |

**Logistic regression (at booking) minus best baseline (Airline delay rate), Brier:** -0.017 [-0.019, -0.015]. The interval excludes 0, so the model is better.

**LightGBM, calibrated (at booking) minus best baseline (Airline delay rate), Brier:** -0.023 [-0.027, -0.021]. The interval excludes 0, so the model is better.

**Logistic regression (1 h before departure) minus best baseline (Airline delay rate), Brier:** -0.058 [-0.063, -0.053]. The interval excludes 0, so the model is better.

**LightGBM, calibrated (1 h before departure) minus best baseline (Airline delay rate), Brier:** -0.072 [-0.077, -0.066]. The interval excludes 0, so the model is better.

**Logistic regression (at booking) minus best baseline (Airline delay rate), PR-AUC:** 0.143 [0.126, 0.160]. The interval excludes 0, so the model is better.

**LightGBM, calibrated (at booking) minus best baseline (Airline delay rate), PR-AUC:** 0.205 [0.184, 0.230]. The interval excludes 0, so the model is better.

**Logistic regression (1 h before departure) minus best baseline (Airline delay rate), PR-AUC:** 0.369 [0.348, 0.392]. The interval excludes 0, so the model is better.

**LightGBM, calibrated (1 h before departure) minus best baseline (Airline delay rate), PR-AUC:** 0.443 [0.423, 0.464]. The interval excludes 0, so the model is better.

## Decision threshold

The API flags a flight as likely delayed when its calibrated probability reaches a threshold chosen on the out-of-fold predictions as the lowest one with precision of at least 75% (so recall is as high as the precision target allows).

| Model | Threshold | Out-of-fold precision | Out-of-fold recall | Test precision | Test recall | Flagged in test |
| --- | --- | --- | --- | --- | --- | --- |
| LightGBM, calibrated (at booking) | 0.63 | 0.750 | 0.032 | 0.875 [0.802, 0.941] | 0.043 [0.033, 0.054] | 1% |
| LightGBM, calibrated (1 h before departure) | 0.49 | 0.750 | 0.398 | 0.810 [0.785, 0.836] | 0.469 [0.443, 0.491] | 15% |

## Calibration

Test flights grouped by predicted probability (at booking). For a calibrated model the observed share delayed matches the predicted probability. Calibration is sigmoid scaling fit on 94952 out-of-fold predictions from rolling-origin CV over train+val. Chart: `reliability.png`.

| Predicted (mean) | Observed | Flights | | Model |
| --- | --- | --- | --- | --- |
| 0.09 | 0.06 | 172 | | LightGBM, calibrated |
| 0.15 | 0.15 | 2992 | | LightGBM, calibrated |
| 0.24 | 0.29 | 1860 | | LightGBM, calibrated |
| 0.34 | 0.40 | 734 | | LightGBM, calibrated |
| 0.45 | 0.61 | 277 | | LightGBM, calibrated |
| 0.54 | 0.70 | 107 | | LightGBM, calibrated |
| 0.64 | 0.85 | 59 | | LightGBM, calibrated |
| 0.75 | 0.83 | 24 | | LightGBM, calibrated |
| 0.85 | 0.88 | 16 | | LightGBM, calibrated |
| 0.09 | 0.08 | 379 | | LightGBM, uncalibrated |
| 0.15 | 0.15 | 2814 | | LightGBM, uncalibrated |
| 0.24 | 0.29 | 1668 | | LightGBM, uncalibrated |
| 0.35 | 0.37 | 729 | | LightGBM, uncalibrated |
| 0.44 | 0.53 | 333 | | LightGBM, uncalibrated |
| 0.54 | 0.67 | 168 | | LightGBM, uncalibrated |
| 0.64 | 0.75 | 71 | | LightGBM, uncalibrated |
| 0.74 | 0.89 | 46 | | LightGBM, uncalibrated |
| 0.85 | 0.85 | 26 | | LightGBM, uncalibrated |
| 0.92 | 0.86 | 7 | | LightGBM, uncalibrated |
| 0.10 | 0.18 | 17 | | Logistic regression |
| 0.16 | 0.12 | 2305 | | Logistic regression |
| 0.24 | 0.29 | 2825 | | Logistic regression |
| 0.34 | 0.41 | 721 | | Logistic regression |
| 0.44 | 0.59 | 319 | | Logistic regression |
| 0.53 | 0.68 | 53 | | Logistic regression |
| 0.66 | 1.00 | 1 | | Logistic regression |

Expected calibration error on test: LightGBM, calibrated 0.037, LightGBM, uncalibrated 0.028, Logistic regression 0.049.

## Error analysis

Test errors by segment. MAE and Brier for the model at booking, MAE also for the model 1 h before departure; coverage is of the conformal 10-90% range. Segments are small, so differences of a few minutes are within noise.

| Segment | Value | Flights | Delayed | MAE (at booking) | MAE (1 h before departure) | Coverage | Brier |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Airport | ATL | 6241 | 26% | 18.4 | 14.3 | 79% | 0.171 |
| Time of day | Morning | 2160 | 13% | 10.2 | 9.0 | 83% | 0.101 |
| Time of day | Afternoon | 1808 | 22% | 15.6 | 13.0 | 82% | 0.156 |
| Time of day | Evening | 1436 | 41% | 28.7 | 21.7 | 74% | 0.240 |
| Time of day | Night | 837 | 42% | 27.5 | 18.1 | 71% | 0.268 |
| Airline | DL | 4817 | 23% | 15.9 | 12.7 | 78% | 0.162 |
| Airline | F9 | 479 | 38% | 30.1 | 21.4 | 78% | 0.221 |
| Airline | other airlines | 344 | 32% | 28.1 | 19.8 | 79% | 0.210 |
| Airline | WN | 301 | 43% | 21.0 | 19.0 | 88% | 0.170 |
| Airline | UA | 157 | 25% | 17.2 | 12.6 | 87% | 0.178 |
| Airline | AA | 143 | 41% | 34.1 | 23.0 | 83% | 0.212 |
| Inbound flight linked | yes | 5158 | 27% | 18.1 | 13.2 | 79% | 0.175 |
| Inbound flight linked | no | 1083 | 24% | 19.5 | 19.5 | 80% | 0.154 |
| Previous delay of aircraft known | yes | 5315 | 25% | 16.0 | 12.4 | 80% | 0.168 |
| Previous delay of aircraft known | no | 926 | 29% | 31.9 | 25.3 | 72% | 0.187 |

## What drives the predictions

Mean absolute SHAP value of the uncalibrated LightGBM classifier on test, in log-odds of a delay over 15 minutes, summed over each feature's one-hot columns. Calibration is monotone, so it does not change the ranking. Chart: `shap_importance.png`.

| Feature | Mean \|SHAP\| |
| --- | --- |
| `Scheduled_Hour_of_Day` | 0.263 |
| `airline_delay_rate_7d` | 0.195 |
| `scheduled_turnaround_min` | 0.179 |
| `has_inbound` | 0.150 |
| `route_median_delay_7d` | 0.135 |
| `airline` | 0.127 |
| `scheduled_day_of_week` | 0.119 |
| `nth_flight_of_day_for_aircraft` | 0.065 |
| `destination_or_origin` | 0.064 |
| `scheduled_time_of_day` | 0.035 |
| `scheduled_minute` | 0.018 |
| `is_weekend` | 0.017 |

## Feature ablation

Each row adds one feature group to the row above (LightGBM median regression for minutes, logistic regression for the classifier, default parameters, fit on train+val). Changes are paired differences against the first row; negative is better for MAE and Brier.

| Features | MAE (min) | Change in MAE | Brier | Change in Brier | PR-AUC |
| --- | --- | --- | --- | --- | --- |
| calendar + identity | 19.3 [18.4, 20.2] | reference | 0.185 [0.178, 0.191] | reference | 0.403 [0.381, 0.426] |
| + rotation | 18.7 [17.8, 19.6] | -0.6 [-0.7, -0.5] | 0.184 [0.178, 0.190] | -0.000 [-0.001, +0.000] | 0.407 [0.386, 0.430] |
| + congestion | 19.1 [18.2, 20.0] | -0.2 [-0.5, +0.1] | 0.184 [0.177, 0.190] | -0.001 [-0.001, -0.000] | 0.406 [0.385, 0.429] |
| + rates | 18.6 [17.8, 19.5] | -0.7 [-0.9, -0.4] | 0.179 [0.173, 0.184] | -0.006 [-0.007, -0.005] | 0.424 [0.401, 0.448] |
| + live_propagation | 14.5 [13.8, 15.3] | -4.8 [-5.3, -4.4] | 0.137 [0.131, 0.142] | -0.048 [-0.053, -0.043] | 0.662 [0.638, 0.687] |

## Feature coverage

BTS records the tail number of nearly every flight, so rotation and live features exist for most departures; they are missing for an aircraft's first flight of the day. Rates are empty for the first week of data.

| Group | Feature | Known in train+val | Known in test |
| --- | --- | --- | --- |
| rotation | `has_inbound` | 100% | 100% |
| rotation | `scheduled_turnaround_min` | 80% | 83% |
| rotation | `nth_flight_of_day_for_aircraft` | 100% | 100% |
| live_propagation | `prev_departure_delay_min` | 83% | 85% |
| live_propagation | `inbound_delay_min` | 60% | 55% |
| live_propagation | `inbound_overdue_min` | 20% | 27% |
| rates | `airline_delay_rate_7d` | 99% | 100% |
| rates | `route_median_delay_7d` | 99% | 100% |

## Choices made from out-of-fold predictions

- at booking: target transform **none** (out-of-fold MAE: none 15.04, log 15.07, clip_300 15.04); conformal margin +0.7 min; threshold 0.63.
- 1 h before departure: target transform **none** (out-of-fold MAE: none 12.29, log 12.30, clip_300 12.29); conformal margin +0.9 min; threshold 0.49.

Hyperparameters were chosen by rolling-origin cross-validation on train+val:

- at booking: CV MAE 15.0 min, CV Brier 0.144 (LightGBM) and 0.150 (logistic regression)
- 1 h before departure: CV MAE 12.3 min, CV Brier 0.110 (LightGBM) and 0.122 (logistic regression)

## Data cleaning

Rows remaining after each cleaning step:

| Step | Rows |
| --- | --- |
| BTS flights touching ATL | 256,833 |
| Departure and arrival rows | 513,666 |
| Domestic departures from ATL | 128,430 |
| ... with status departed | 125,032 |
| ... with a departure time | 125,032 |
| ... with a delay in [-60, 720] min | 124,931 |

## Lineage

| Field | Value |
| --- | --- |
| git_commit | `f253f48f80292332d9754fc656ee31fc1e114f41` |
| git_dirty | `false` |
| raw_data_sha256 | `8ee3c5d7ddc8fc4b08ff408997672a52ace79493a389e637a8387515e6f0c013` |
| training_data_sha256 | `a8a3e3aae2e1b3949e3a061a88bfbb784a39a3ce143c2b85726bf5070f3922f3` |
| training_rows | `118690` |
| test_rows | `6241` |
| training_from | `2026-01-01 05:10:00` |
| training_to | `2026-05-24 23:20:00` |
| test_from | `2026-05-25 05:00:00` |
| test_to | `2026-05-31 23:23:00` |
