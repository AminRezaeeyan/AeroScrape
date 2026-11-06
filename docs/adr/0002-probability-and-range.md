# 2. Predict a calibrated probability and a delay range

**Status:** accepted, November 2026

## Context

Delays are skewed: in the 2025 training data the median delay is 20 minutes, one flight in ten is more than 75 minutes late and one in a hundred more than four hours. A model that predicts minutes is wrong for most flights by a lot, and its error hides how uncertain it is.

## Decision

Return three things: the calibrated probability of a delay over 15 minutes (LightGBM, Platt scaling fit on out-of-fold predictions), a 10th–90th percentile range from quantile LightGBM models, widened by a conformal margin so it really holds about 80% of flights, and a "likely delayed" flag set where out-of-fold precision reaches 75%. They are packaged as one MLflow model, so they can't get out of step.

## Alternatives

- **Point estimate of minutes:** see above.
- **Isotonic calibration:** overfits a few thousand out-of-fold predictions; Platt scaling is smoother.
- **Raw quantiles:** covered 79% instead of the target 80% on test before the conformal margin, and less in some segments.
- **A 0.5 cutoff for the flag:** with 60% of flights delayed, it flags most flights.

## Consequences

Reliability diagrams and coverage are part of the evaluation and of live monitoring (Brier score and coverage per day), and the gate checks both.
