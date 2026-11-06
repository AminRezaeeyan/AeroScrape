# 6. Compute input drift in the project instead of with Evidently

**Status:** accepted, November 2026

## Context

The monitor DAG compares the last 7 days of model inputs with the training inputs for a handful of features and needs a number per feature, a threshold and a readable report.

## Decision

The Population Stability Index per feature (deciles of the training values for numbers, values for categories and flags, missing values as their own bin), the Kolmogorov–Smirnov p-value for numbers, and a self-contained HTML report, in `aeroscrape/monitoring/drift.py` with unit tests. The reference sample is rebuilt from the model's training window with the training code and stored with its MLflow run.

## Alternatives

- **Evidently:** the plan's suggestion. It adds dozens of dependencies (and telemetry) to the Airflow image, for the same numbers.
- **KS only:** with thousands of rows it flags tiny shifts; PSI has accepted thresholds (0.1, 0.25).

## Consequences

About 150 lines to maintain. Adding a test (e.g. Jensen–Shannon distance) is a function, not an upgrade.
