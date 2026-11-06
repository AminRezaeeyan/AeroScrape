# Learning curve (bts)

Each row trains on the given number of weeks immediately before the same test window, with default parameters and the 1-hour-before features. The change is LightGBM minus the airline × time-of-day baseline trained on the same weeks, with a paired 95% bootstrap interval. Chart: `learning_curve.png`.

| Weeks | Training departures | LightGBM MAE | Baseline MAE | Change vs baseline |
| --- | --- | --- | --- | --- |
| 1 | 6,021 | 15.66 | 18.85 | -3.19 [-3.57, -2.86] |
| 2 | 12,230 | 14.55 | 18.93 | -4.38 [-4.84, -3.99] |
| 4 | 24,432 | 14.60 | 19.11 | -4.50 [-4.92, -4.13] |
| 8 | 48,870 | 14.48 | 19.44 | -4.96 [-5.41, -4.56] |
| 16 | 94,544 | 14.56 | 19.32 | -4.77 [-5.21, -4.36] |
