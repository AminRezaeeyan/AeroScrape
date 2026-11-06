# Running the pipeline on US BTS data

Three weeks of Iranian data can't show how the pipeline behaves at scale. The US Bureau of Transportation Statistics publishes every US domestic flight, by month, with the aircraft's tail number, so the same delay-propagation features apply. Running the identical pipeline on it shows that the pipeline is source-agnostic and how results change with data volume.

## How BTS maps to the pipeline

[`scraping/bts.py`](../src/aeroscrape/scraping/bts.py) downloads the monthly "Reporting Carrier On-Time Performance" zips from TranStats and translates them into the cleaned flight schema. Everything after that, from features and splits to the models, the gate and the report, is unchanged.

| Cleaned column | BTS column |
| --- | --- |
| `scheduled_datetime` (departure) | `FlightDate` + `CRSDepTime` (2400 is midnight) |
| `actual_datetime` (departure) | scheduled + `DepDelay` |
| `scheduled_datetime` (arrival) | `FlightDate` + `CRSArrTime`, next day if it is over 2 hours before the departure clock time |
| `airline`, `flight_number` | `Reporting_Airline`, airline + `Flight_Number_Reporting_Airline` |
| `airport`, `destination_or_origin` | `Origin` / `Dest` |
| `register` | `Tail_Number` (invalid values dropped) |
| `status` | `Cancelled` → cancelled, `Diverted` → unknown, otherwise departed or landed |

Each BTS flight becomes a departure row at its origin and an arrival row at its destination, the way the Iranian boards list a flight once at each end. BTS has no aircraft type, so that feature is empty.

## Configuration

`config.yaml` has a `profiles.bts` section that overrides the defaults when `SOURCE=bts`:

- **Data:** January to May 2026 for flights touching **Atlanta (ATL)**. Only ATL departures are predicted; flights at other airports are context for rotation features.
- **Held-out window:** 7 test dates and 7 validation dates, against 3 and 2 for the Iranian data.
- **Lighter tuning,** since the data is about 50 times larger.
- **A separate registry name,** `flight-delay-model-bts`, so the gate never compares a US model with an Iranian one.

## Running it

BTS blocks some networks and VPN exits (a TCP timeout to `transtats.bts.gov`, or 403 from `bts.gov`). With access:

```bash
make bts-data                  # download Jan-May 2026 into data/bts/ (~5 x 25 MB)
make evaluate SOURCE=bts       # tune, train, calibrate, evaluate; writes reports/bts/
make learning-curve            # test MAE for 1, 2, 4, 8 and 16 weeks of training data
make train SOURCE=bts          # the same through Airflow, with the gate and the BTS registry
```

`make evaluate SOURCE=bts` takes on the order of 30-60 minutes on a laptop, most of it tuning. If the download fails, save these files in a browser into `data/bts/` under the names shown, and the commands above will use them:

| Month | URL | Save as |
| --- | --- | --- |
| 2026-01 | https://transtats.bts.gov/PREZIP/On_Time_Reporting_Carrier_On_Time_Performance_1987_present_2026_1.zip | `data/bts/bts_2026_01.zip` |
| 2026-02 | https://transtats.bts.gov/PREZIP/On_Time_Reporting_Carrier_On_Time_Performance_1987_present_2026_2.zip | `data/bts/bts_2026_02.zip` |
| 2026-03 | https://transtats.bts.gov/PREZIP/On_Time_Reporting_Carrier_On_Time_Performance_1987_present_2026_3.zip | `data/bts/bts_2026_03.zip` |
| 2026-04 | https://transtats.bts.gov/PREZIP/On_Time_Reporting_Carrier_On_Time_Performance_1987_present_2026_4.zip | `data/bts/bts_2026_04.zip` |
| 2026-05 | https://transtats.bts.gov/PREZIP/On_Time_Reporting_Carrier_On_Time_Performance_1987_present_2026_5.zip | `data/bts/bts_2026_05.zip` |

## The learning curve

[`models/learning_curve.py`](../src/aeroscrape/models/learning_curve.py) trains on the last 1, 2, 4, 8 and 16 weeks before the same test week, with default parameters, and plots test MAE for the model and the airline × time-of-day baseline against the number of training departures. A dashed line marks the size of the Iranian training set (2,887 departures), so the curve answers "how much Iranian data would the model need?" with evidence.

## Status

The adapter, profile, learning curve and Airflow parameter are tested against synthetic data in the exact BTS file layout (`tests/bts_synthetic.py`, `tests/test_bts.py`), including a full pipeline run. **Results on real BTS data are pending**: BTS was not reachable from the development machine, so they need `make bts-data` from a network that can reach it.

## Limits

BTS is US data: US hubs, carriers, weather and air-traffic control behave differently from Iranian airports, so its numbers show what the pipeline can do with enough data, not what the Iranian model will reach. Delays beyond 12 hours are treated as data errors in both sources (a small share of BTS flights).
