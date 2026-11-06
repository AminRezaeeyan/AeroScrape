# 1. Split by time, tune inside the training window

**Status:** accepted, November 2026

## Context

The first version split flights at random and chose hyperparameters on the test set. Flights on the same day share weather, congestion and knock-on delays, so a random split lets the model see the test day's conditions through its neighbours, and tuning on test turns the test score into a training score.

## Decision

Split by scheduled time. The test set is the newest 3 dates with data, validation the 2 before, training everything earlier (`evaluation` in `config.yaml`). Hyperparameters, calibration, the target transform and the decision threshold are chosen by rolling-origin cross-validation (`TimeSeriesSplit`) on train+val. The test set is scored once. Every result is reported against baselines with paired bootstrap intervals.

## Alternatives

- **Random split:** optimistic for the reason above.
- **One fixed date cut:** reproducible, but the test window never moves as new flights are scraped. The relative cut keeps testing on the newest data; set `train_end` and `val_end` to fix it.
- **Grouped split by day:** removes same-day leakage but still trains on the future.

## Consequences

Numbers are lower than the first version's and honest. With three weeks of data the test set is small (687 flights), so intervals are wide and only some differences are significant.
