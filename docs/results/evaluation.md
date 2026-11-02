<!-- Snapshot of reports/evaluation.md from `make evaluate` on data/raw/data.csv (sha256 7b9ba655...). Regenerate after changing data, features or models. -->

# Evaluation report

Test window: 2025-06-10 to 2025-06-12 (442 flights). Models and baselines are fit on train+val and scored once on test. Brackets are 95% bootstrap percentile intervals over 1,000 resamples of the test set; differences are paired, using the same resamples for both models.

## Data

| Split | Rows | From | To |
| --- | --- | --- | --- |
| train | 1551 | 2025-05-22 00:45 | 2025-06-04 23:30 |
| val | 320 | 2025-06-06 04:30 | 2025-06-09 23:30 |
| test | 442 | 2025-06-10 04:40 | 2025-06-12 23:45 |

## Delay in minutes (regression)

Test delays: median 23 min, mean 37 min, max 537 min.

| Model | MAE (min) | MedAE (min) | Within 15 min |
| --- | --- | --- | --- |
| Global median (baseline) | 25.2 [20.9, 29.8] | 12.0 [10.0, 12.0] | 64.9% [60.2%, 69.2%] |
| Airline × time-of-day median (baseline) | 24.6 [20.3, 29.3] | 11.0 [10.0, 12.0] | 64.0% [59.7%, 68.6%] |
| **LightGBM** | 24.4 [20.2, 29.0] | 11.6 [10.5, 12.6] | 63.1% [58.8%, 67.6%] |

**LightGBM minus best baseline (Airline × time-of-day median), MAE (min):** -0.2 [-0.8, 0.2]. The interval includes 0, so this test set gives no evidence of a difference.

## Delayed more than 15 minutes (classification)

67.9% of test flights are delayed, so always predicting 'delayed' scores 0.679 accuracy. PR-AUC of a random ranking equals that rate.

| Model | Accuracy | PR-AUC | Brier |
| --- | --- | --- | --- |
| Majority class (baseline) | 0.679 [0.636, 0.722] | 0.679 [0.636, 0.722] | 0.219 [0.206, 0.232] |
| Airline delay rate (baseline) | 0.704 [0.663, 0.744] | 0.781 [0.728, 0.832] | 0.202 [0.187, 0.218] |
| **Logistic regression** | 0.697 [0.654, 0.740] | 0.797 [0.741, 0.847] | 0.202 [0.186, 0.220] |

**Logistic regression minus best baseline (Airline delay rate), Brier:** 0.000 [-0.008, 0.008]. The interval includes 0, so this test set gives no evidence of a difference.

## Tuning

Parameters were chosen by rolling-origin cross-validation on train+val (test unused): CV MAE 22.8 min, CV Brier 0.219.
