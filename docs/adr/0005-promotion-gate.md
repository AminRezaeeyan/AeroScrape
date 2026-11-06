# 5. Promote only through a champion/challenger gate

**Status:** accepted, November 2026

## Context

Training runs on a schedule and on alerts, on data that changes with every scrape. A bad scrape or a bug can produce a worse model, and nobody reviews each run.

## Decision

A new model is registered with the alias `challenger`. The gate scores it and the `production` model on the same newest held-out flights and promotes it only if there are enough flights (200), its MAE is better than the champion's and the baseline's with paired 95% bootstrap intervals entirely below zero, its Brier score is at most 0.005 worse, and its range covers 70–90% of flights. Each decision is an MLflow run, and `make rollback VERSION=n` moves the alias back with a recorded reason.

## Alternatives

- **Always promote the newest:** see above.
- **Promote when the point MAE is lower:** with a few hundred test flights, a lower MAE is often noise.
- **Manual promotion:** in practice it doesn't happen.

## Consequences

The gate is conservative: it rejected a retrain on the same data plus one day (no evidence it was better), a model trained on 10% of the data, and the retrain triggered by drift in November 2026. A clearly better model needs a few weeks of new data.
