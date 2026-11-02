<!-- Snapshot of reports/evaluation.md from `make evaluate` on data/raw/data.csv (sha256 7b9ba655...). Charts: reliability.png, shap_importance.png. Regenerate after changing data, features or models. -->

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
| **LightGBM median (at booking)** | 24.0 [20.6, 27.5] | 11.3 [10.5, 12.6] | 61.1% [57.6%, 64.9%] |
| **LightGBM median (1 h before departure)** | 23.0 [19.9, 26.5] | 11.5 [10.6, 12.9] | 61.0% [57.2%, 64.6%] |

**LightGBM median (at booking) minus best baseline (Airline × time-of-day median), MAE (min):** -0.7 [-1.2, -0.3]. The interval excludes 0, so the model is better.

**LightGBM median (1 h before departure) minus best baseline (Airline × time-of-day median), MAE (min):** -1.7 [-2.6, -0.9]. The interval excludes 0, so the model is better.

## Delay range: 10th to 90th percentile

A calibrated 10-90% interval contains about 80% of actual delays. Pinball loss scores all three percentiles at once and rewards narrow intervals that still cover.

| Model | Coverage of 10-90% interval | Mean width (min) | Pinball loss (min) |
| --- | --- | --- | --- |
| Global percentiles (baseline) | 78.7% [75.4%, 82.0%] | 72.0 [72.0, 72.0] | 9.20 [7.78, 10.79] |
| Airline × time-of-day percentiles (baseline) | 76.4% [73.4%, 79.5%] | 68.6 [65.3, 72.0] | 9.19 [7.76, 10.74] |
| **LightGBM percentiles (at booking)** | 78.6% [75.3%, 81.7%] | 70.8 [68.6, 73.1] | 8.80 [7.43, 10.38] |
| **LightGBM percentiles + conformal (at booking)** | 82.7% [79.6%, 85.4%] | 74.6 [72.4, 76.9] | 8.80 [7.44, 10.36] |
| **LightGBM percentiles (1 h before departure)** | 71.5% [68.1%, 74.7%] | 61.1 [58.7, 63.7] | 8.50 [7.15, 10.05] |
| **LightGBM percentiles + conformal (1 h before departure)** | 85.3% [82.7%, 88.1%] | 71.6 [69.3, 74.3] | 8.49 [7.18, 10.03] |

**LightGBM percentiles + conformal (at booking) minus best baseline (Airline × time-of-day percentiles), Pinball loss (min):** -0.39 [-0.67, -0.15]. The interval excludes 0, so the model is better.

**LightGBM percentiles + conformal (1 h before departure) minus best baseline (Airline × time-of-day percentiles), Pinball loss (min):** -0.70 [-1.02, -0.40]. The interval excludes 0, so the model is better.

## Delayed more than 15 minutes (classification)

60.3% of test flights are delayed, so always predicting 'delayed' scores 0.603 accuracy. PR-AUC of a random ranking equals that rate.
Accuracy, precision and recall here use a 0.5 cutoff for every model; the chosen decision threshold is evaluated in its own section below.

| Model | Accuracy | Precision | Recall | PR-AUC | Brier | ECE |
| --- | --- | --- | --- | --- | --- | --- |
| Majority class (baseline) | 0.603 [0.565, 0.640] | 0.603 [0.565, 0.640] | 1.000 [1.000, 1.000] | 0.603 [0.565, 0.640] | 0.240 [0.234, 0.246] | 0.020 [0.001, 0.058] |
| Airline delay rate (baseline) | 0.646 [0.608, 0.681] | 0.655 [0.614, 0.693] | 0.874 [0.843, 0.905] | 0.717 [0.670, 0.762] | 0.221 [0.210, 0.231] | 0.049 [0.026, 0.089] |
| **Logistic regression (at booking)** | 0.684 [0.649, 0.719] | 0.698 [0.657, 0.740] | 0.838 [0.802, 0.874] | 0.754 [0.705, 0.801] | 0.209 [0.198, 0.221] | 0.029 [0.026, 0.074] |
| **LightGBM, uncalibrated (at booking)** | 0.683 [0.648, 0.718] | 0.681 [0.642, 0.720] | 0.889 [0.857, 0.920] | 0.765 [0.715, 0.815] | 0.209 [0.198, 0.220] | 0.051 [0.032, 0.090] |
| **LightGBM, calibrated (at booking)** | 0.705 [0.668, 0.738] | 0.716 [0.675, 0.757] | 0.845 [0.810, 0.881] | 0.765 [0.715, 0.815] | 0.207 [0.196, 0.219] | 0.069 [0.043, 0.104] |
| **Logistic regression (1 h before departure)** | 0.671 [0.636, 0.706] | 0.688 [0.646, 0.729] | 0.831 [0.794, 0.868] | 0.774 [0.729, 0.815] | 0.208 [0.197, 0.219] | 0.039 [0.029, 0.077] |
| **LightGBM, uncalibrated (1 h before departure)** | 0.709 [0.674, 0.741] | 0.723 [0.682, 0.762] | 0.838 [0.802, 0.873] | 0.803 [0.763, 0.842] | 0.203 [0.192, 0.213] | 0.070 [0.043, 0.103] |
| **LightGBM, calibrated (1 h before departure)** | 0.706 [0.671, 0.739] | 0.726 [0.685, 0.765] | 0.824 [0.788, 0.861] | 0.803 [0.763, 0.842] | 0.201 [0.189, 0.212] | 0.053 [0.031, 0.089] |

**Logistic regression (at booking) minus best baseline (Airline delay rate), Brier:** -0.011 [-0.021, -0.003]. The interval excludes 0, so the model is better.

**LightGBM, calibrated (at booking) minus best baseline (Airline delay rate), Brier:** -0.014 [-0.023, -0.005]. The interval excludes 0, so the model is better.

**Logistic regression (1 h before departure) minus best baseline (Airline delay rate), Brier:** -0.013 [-0.021, -0.005]. The interval excludes 0, so the model is better.

**LightGBM, calibrated (1 h before departure) minus best baseline (Airline delay rate), Brier:** -0.020 [-0.030, -0.011]. The interval excludes 0, so the model is better.

**Logistic regression (at booking) minus best baseline (Airline delay rate), PR-AUC:** 0.037 [-0.001, 0.077]. The interval includes 0, so this test set gives no evidence of a difference.

**LightGBM, calibrated (at booking) minus best baseline (Airline delay rate), PR-AUC:** 0.048 [0.009, 0.093]. The interval excludes 0, so the model is better.

**Logistic regression (1 h before departure) minus best baseline (Airline delay rate), PR-AUC:** 0.058 [0.022, 0.090]. The interval excludes 0, so the model is better.

**LightGBM, calibrated (1 h before departure) minus best baseline (Airline delay rate), PR-AUC:** 0.086 [0.051, 0.124]. The interval excludes 0, so the model is better.

## Decision threshold

The API flags a flight as likely delayed when its calibrated probability reaches a threshold chosen on the out-of-fold predictions as the lowest one with precision of at least 75% (so recall is as high as the precision target allows).

| Model | Threshold | Out-of-fold precision | Out-of-fold recall | Test precision | Test recall | Flagged in test |
| --- | --- | --- | --- | --- | --- | --- |
| LightGBM, calibrated (at booking) | 0.69 | 0.751 | 0.303 | 0.795 [0.729, 0.855] | 0.329 [0.284, 0.375] | 25% |
| LightGBM, calibrated (1 h before departure) | 0.69 | 0.750 | 0.324 | 0.845 [0.792, 0.898] | 0.355 [0.312, 0.399] | 25% |

## Calibration

Test flights grouped by predicted probability (at booking). For a calibrated model the observed share delayed matches the predicted probability. Calibration is sigmoid scaling fit on 2308 out-of-fold predictions from rolling-origin CV over train+val. Chart: `reliability.png`.

| Predicted (mean) | Observed | Flights | | Model |
| --- | --- | --- | --- | --- |
| 0.17 | 0.26 | 23 | | LightGBM, calibrated |
| 0.25 | 0.15 | 13 | | LightGBM, calibrated |
| 0.34 | 0.16 | 32 | | LightGBM, calibrated |
| 0.46 | 0.39 | 130 | | LightGBM, calibrated |
| 0.56 | 0.65 | 142 | | LightGBM, calibrated |
| 0.65 | 0.70 | 195 | | LightGBM, calibrated |
| 0.74 | 0.79 | 118 | | LightGBM, calibrated |
| 0.83 | 0.85 | 34 | | LightGBM, calibrated |
| 0.19 | 0.00 | 4 | | LightGBM, uncalibrated |
| 0.24 | 0.25 | 24 | | LightGBM, uncalibrated |
| 0.36 | 0.21 | 28 | | LightGBM, uncalibrated |
| 0.48 | 0.37 | 91 | | LightGBM, uncalibrated |
| 0.55 | 0.57 | 184 | | LightGBM, uncalibrated |
| 0.65 | 0.70 | 218 | | LightGBM, uncalibrated |
| 0.74 | 0.79 | 120 | | LightGBM, uncalibrated |
| 0.83 | 0.89 | 18 | | LightGBM, uncalibrated |
| 0.19 | 0.00 | 3 | | Logistic regression |
| 0.26 | 0.18 | 22 | | Logistic regression |
| 0.36 | 0.25 | 56 | | Logistic regression |
| 0.45 | 0.45 | 109 | | Logistic regression |
| 0.56 | 0.63 | 107 | | Logistic regression |
| 0.65 | 0.65 | 182 | | Logistic regression |
| 0.75 | 0.76 | 160 | | Logistic regression |
| 0.84 | 0.83 | 47 | | Logistic regression |
| 0.92 | 1.00 | 1 | | Logistic regression |

Expected calibration error on test: LightGBM, calibrated 0.069, LightGBM, uncalibrated 0.051, Logistic regression 0.029.

## Error analysis

Test errors by segment. MAE and Brier for the model at booking, MAE also for the model 1 h before departure; coverage is of the conformal 10-90% range. Segments are small, so differences of a few minutes are within noise.

| Segment | Value | Flights | Delayed | MAE (at booking) | MAE (1 h before departure) | Coverage | Brier |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Airport | فرودگاه مهرآباد | 413 | 68% | 24.7 | 23.2 | 83% | 0.198 |
| Airport | other airports | 245 | 47% | 23.3 | 23.2 | 83% | 0.219 |
| Airport | فرودگاه مشهد | 29 | 62% | 19.4 | 19.3 | 79% | 0.229 |
| Time of day | Evening | 208 | 59% | 23.0 | 21.8 | 83% | 0.202 |
| Time of day | Morning | 200 | 67% | 24.0 | 24.1 | 81% | 0.217 |
| Time of day | Afternoon | 145 | 57% | 25.0 | 22.7 | 86% | 0.207 |
| Time of day | Night | 134 | 57% | 24.3 | 23.5 | 81% | 0.200 |
| Airline | other airlines | 383 | 57% | 24.6 | 24.0 | 81% | 0.219 |
| Airline | آتا | 104 | 79% | 31.3 | 28.3 | 87% | 0.164 |
| Airline | ماهان | 69 | 33% | 11.0 | 11.4 | 86% | 0.196 |
| Airline | ایران ایر | 48 | 69% | 18.5 | 17.8 | 85% | 0.221 |
| Airline | ایران ایرتور | 44 | 73% | 32.5 | 29.2 | 84% | 0.204 |
| Airline | کیش ایر | 39 | 62% | 18.7 | 19.5 | 82% | 0.209 |
| Inbound flight linked | no | 412 | 57% | 25.8 | 25.6 | 82% | 0.215 |
| Inbound flight linked | yes | 275 | 65% | 21.3 | 19.2 | 83% | 0.195 |
| Previous delay of aircraft known | no | 463 | 57% | 25.0 | 25.0 | 83% | 0.213 |
| Previous delay of aircraft known | yes | 224 | 66% | 21.9 | 18.9 | 83% | 0.194 |

## What drives the predictions

Mean absolute SHAP value of the uncalibrated LightGBM classifier on test, in log-odds of a delay over 15 minutes, summed over each feature's one-hot columns. Calibration is monotone, so it does not change the ranking. Chart: `shap_importance.png`.

| Feature | Mean \|SHAP\| |
| --- | --- |
| `nth_flight_of_day_for_aircraft` | 0.210 |
| `airline` | 0.134 |
| `airline_delay_rate_7d` | 0.099 |
| `route_median_delay_7d` | 0.095 |
| `scheduled_turnaround_min` | 0.070 |
| `Scheduled_Hour_of_Day` | 0.048 |
| `destination_or_origin` | 0.045 |
| `scheduled_day_of_week` | 0.042 |
| `aircraft` | 0.038 |
| `airport` | 0.030 |
| `scheduled_minute` | 0.023 |
| `scheduled_time_of_day` | 0.016 |

## Feature ablation

Each row adds one feature group to the row above (LightGBM median regression for minutes, logistic regression for the classifier, default parameters, fit on train+val). Changes are paired differences against the first row; negative is better for MAE and Brier.

| Features | MAE (min) | Change in MAE | Brier | Change in Brier | PR-AUC |
| --- | --- | --- | --- | --- | --- |
| calendar + identity | 23.8 [20.6, 27.2] | reference | 0.209 [0.196, 0.224] | reference | 0.763 [0.716, 0.810] |
| + rotation | 23.8 [20.6, 27.2] | -0.0 [-0.3, +0.3] | 0.209 [0.196, 0.224] | +0.000 [-0.003, +0.002] | 0.764 [0.716, 0.810] |
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

## Choices made from out-of-fold predictions

- at booking: target transform **none** (out-of-fold MAE: none 24.06, log 24.06, clip_300 24.06); conformal margin +1.9 min; threshold 0.69.
- 1 h before departure: target transform **none** (out-of-fold MAE: none 23.74, log 23.78, clip_300 23.74); conformal margin +5.3 min; threshold 0.69.

Hyperparameters were chosen by rolling-origin cross-validation on train+val:

- at booking: CV MAE 24.1 min, CV Brier 0.222 (LightGBM) and 0.224 (logistic regression)
- 1 h before departure: CV MAE 23.7 min, CV Brier 0.222 (LightGBM) and 0.224 (logistic regression)

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
