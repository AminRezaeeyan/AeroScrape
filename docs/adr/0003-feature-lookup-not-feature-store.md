# 3. Build serving features from the flight table, with the training code

**Status:** accepted, November 2026

## Context

The useful features (the aircraft's previous delay, its inbound flight, the airline's delays over the last week) depend on other flights. The first API had none of them and filled every one with a default, so served predictions were weaker than tested ones.

## Decision

The API fetches the flights a request's features look at from Postgres (the days before for rates, the hours before for the aircraft's rotation), cleans them with the training cleaner, and calls the same `build_features` that training uses. Earlier days are cached per date; the flight's own day is fetched on every request so live delays are current. A parity test builds features for 50 flights through both paths and requires identical values.

## Alternatives

- **Feast or another feature store:** a registry, an online store and materialization jobs for about 600 flights a day.
- **A snapshot table refreshed by Airflow:** simpler to read, but live delays would be as old as the last refresh, and it is a second implementation of every feature.
- **Features in the request:** clients don't know an aircraft's previous delay.

## Consequences

A prediction costs about 100 ms, most of it the lookup. If traffic grows, a snapshot refreshed by the ingest DAG is the next step, still built by the same function.
