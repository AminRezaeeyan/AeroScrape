# Design

## Context

Iran Airports Company publishes a flight information display (FIDS) page for each airport: about a
day of arrivals and departures with their status, and for some flights the aircraft registration.
Nobody publishes delay history. AeroScrape builds that history by scraping the boards, and
predicts, for one domestic departure, how likely a delay over 15 minutes is and how long it is
likely to be.

The users are travellers and the people who plan around a flight. They decide whether to leave
for the airport later or rebook a connection. That decision needs an honest probability and a
range, not a single number of minutes that will be wrong for most flights.

## Goals

- A calibrated probability of a delay over 15 minutes, and a 10–90% delay range that really
  holds about 8 in 10 flights.
- Two moments: at booking, and one hour before departure, when the aircraft's previous delay is
  known.
- Numbers measured on future flights, against simple baselines, with intervals.
- A system that keeps working unattended: it scrapes, trains, gates, serves and notices when
  accuracy or inputs change.

## Non-goals

- Real-time streaming. The boards change every few minutes, but scraping every three hours
  captures final statuses, and a delay forecast an hour ahead doesn't need seconds.
- International flights, and arrivals as prediction targets. Arrivals are used as features.
- High availability. One API worker on one machine serves about 10 single predictions or
  340 flights in batches per second, far beyond the traffic.
- Explaining delays causally. SHAP shows what the model uses, not why flights are late.

## Key decisions

Each has a short record with the alternatives in [adr/](adr/).

| Decision | Chosen | Rejected, and why |
| --- | --- | --- |
| Evaluation split | By scheduled time: test is the newest 3 days, validation the 2 before, tuning by rolling-origin CV inside train+val ([ADR 1](adr/0001-time-based-split.md)) | A random split leaks the same day's conditions into training. The first version used one and tuned on the test set; its numbers didn't survive an honest split |
| Output | Calibrated probability plus quantile range, with conformal widening ([ADR 2](adr/0002-probability-and-range.md)) | A point estimate of minutes: delays are skewed (median 20 min, 1 in 10 over 75 min), so one number is wrong for most flights and hides the uncertainty |
| Model | LightGBM for the quantiles and the classifier, Platt-scaled on out-of-fold predictions | Deep tabular models: 3,574 training rows, no evidence they beat boosted trees on small tabular data, harder to explain. Logistic regression stays as a comparison |
| Features at serving | A lookup that builds features from the flight table with the training code, per request, with a per-day cache ([ADR 3](adr/0003-feature-lookup-not-feature-store.md)) | Feast or another feature store: an extra service for a few thousand rows a day; precomputed snapshots: stale live features |
| Orchestration | Airflow, three DAGs on their own schedules ([ADR 4](adr/0004-airflow.md)) | Cron: no retries, no history, no dependencies between steps, no UI to see what failed |
| Promotion | A champion/challenger gate on the newest held-out flights, with paired bootstrap intervals ([ADR 5](adr/0005-promotion-gate.md)) | Promote the newest model: a worse model can go live after a bad scrape. Manual promotion: doesn't happen |
| Drift | PSI per input and KS for numbers, about 150 tested lines ([ADR 6](adr/0006-own-drift-checks.md)) | Evidently: dozens of dependencies in the Airflow image for the same numbers |
| Monitoring | Prometheus, Pushgateway and Grafana, configured as code; daily live metrics in Postgres | A hosted service: costs, and the stack must run offline on a laptop |

## Data flow

1. **Ingest** (every 3 hours): fetch each airport's page with retries, save it, parse, validate
   the batch (contract and minimum size), upsert changed flights. A failed page fails the run
   after the others are loaded.
2. **Train** (weekly, or on a monitoring alert): clean every flight from the archive and the
   database, split by time, tune, fit, evaluate against baselines, register a challenger, run the
   gate, promote or reject.
3. **Serve**: the API loads the production version, polls the registry every two minutes and
   swaps models atomically. For each request it builds features from the flight table with the
   training code, scores, and logs the prediction after responding.
4. **Monitor** (daily): match logged predictions to actual delays with the training cleaner,
   compute daily MAE, Brier and coverage against the offline numbers, compare recent inputs with
   the training inputs, and trigger training when needed.

## Failure modes

| Failure | What happens | Mitigation |
| --- | --- | --- |
| The site changes its layout | Tables missing or with unknown headers | `LayoutChangedError` per page; the run fails and `ScrapePageFailed` fires; saved pages can be re-parsed after the fix |
| The site is slow or down | Timeouts | Timeouts and three retries with backoff; a failed page fails the run; `IngestStale` after 7 hours without a load |
| A scrape returns too little | e.g. one airport's table fails to render | Minimum rows per batch and for the busiest airports; nothing is loaded |
| Registrations or inbound data missing | Rotation features empty | The model was trained with missing values and fills them; the API lists each unknown feature in `warnings`; the share is tracked per run and per day |
| A category the model never saw | e.g. a new airline | Accepted (the model ignores it) and reported in `warnings` and `unseen_categories_total` |
| The model degrades | Season, new airlines, schedule changes | Daily live MAE against offline; three degraded days or input drift trigger retraining |
| A bad model gets trained | Bad data, a bug | The gate rejects it; if one is promoted anyway, `make rollback VERSION=n` |
| Train/serve skew | Features computed differently | One feature function, and a parity test on 50 flights |
| The prediction log is down | Database outage | Predictions still return; `PredictionLogFailing` fires |

## With more time or data

- **Weather.** METAR reports for each airport are the obvious missing cause of long delays.
- **A year of history**, for seasonality, Nowruz and the lunar holidays, which shift every year.
- **Registrations everywhere.** The aircraft's previous delay is the strongest feature, and only
  a third of flights have one. Matching aircraft by airline, type and timing might recover some.
- **Airport capacity and runway closures** (NOTAMs).
- **Retrain on recent data.** The served model is from 2025. With a few weeks of continuous
  scraping, the gate's newest held-out window will be from 2026, and a model trained on it should
  win.
