# Data source: fids.airport.ir

AeroScrape reads the public flight information display (FIDS) pages of Iran Airports Company, one page per airport, listed in `config.yaml`. These notes record how the pages are structured and where they are unreliable. Last checked 2026-11-06 against all 34 pages.

## Page structure

Each airport page has up to four tables, each inside a `div.tabcontent` whose `id` names the flight type:

| `div` id | Flight type | Origin or destination header |
| --- | --- | --- |
| `input` | Domestic Arrival | مبدا |
| `output` | Domestic Departure | مقصد |
| `internal` | International Arrival | مبدا |
| `external` | International Departure | مقصد |

Airports without international flights (e.g. Mehrabad) have no `internal` or `external` table.

### Columns

The parser maps cells to fields by **header text**, not by position (`HEADER_FIELDS` in `scraping/scraper.py`), because tables don't share one layout:

- The header row starts with a "day" header (روز) that has no cell of its own. The first cell holds the weekday and scheduled time together, e.g. `دوشنبه 06:20`.
- International departures have no counter column, so their rows have 9 cells instead of 10.
- Departure tables label the actual time column `تاریخ و زمان واقعی`; arrival tables label it `زمان واقعی`.

If a table has an unknown header or a row whose cell count doesn't match its header, the scraper logs `Layout changed on ...` and skips that table. `tests/fixtures/fids_mashhad.html` is a trimmed copy of a real page used by the parser tests.

## Known data quality issues

| Issue | Example | Handling today |
| --- | --- | --- |
| Weekday names vary in spacing | `دوشنبه` on the site, `دو شنبه` in older code | Weekdays are compared after removing spaces and zero-width non-joiners and unifying Arabic and Persian `ی`/`ک` |
| Scheduled time has a weekday, not a date | `دوشنبه 06:20` | The date is the nearest matching weekday to the scrape date (from two days back to two days ahead) |
| Actual time is time only | `06:35` | Placed on the flight's scheduled date; moved to the next day if it is more than 3 hours before the schedule |
| The actual time cell can hold an estimate | CPN024: cell `07:10`, status `تخمین نشست 07:10 \| نشست07:15` | The cleaner takes the time from the status text (`نشست07:15`) and uses the cell only as a fallback, and only when the status says the flight departed or landed |
| Status text embeds times and varies in spelling | `پروازکرد04:57`, older data `پرواز كرد` | Stored as-is in the database; the cleaner maps it to one of a few states (`FlightStatus` in `data/clean.py`). Before this, only `پرواز كرد` was kept, which dropped 1,261 departures and every airport except Mehrabad and Mashhad |
| The date column disagrees with the weekday for some international flights | Monday flight dated `2026-11-04`; one cell reads `m 2026-11-04` | Not used. Domestic rows' dates match their weekday |
| Airline cell can be a logo without text | `<img src=".../IRM.png">` | Falls back to the logo's airline code (`IRM`) |
| The page shows a rolling window | about one day of flights per airport | Scraping only once a day misses flights and final statuses |
| Registrations only at Mehrabad | 55% of Mehrabad departures, almost none elsewhere; 84% of all scraped flights had none on 2026-11-06 | Limits aircraft-rotation features; see [features.md](features.md). Tracked per run as `scrape_null_share_register` |
| Small airports' boards are often empty | Sari and Hamedan at night: tables with headers and no rows | Normal, not an error. 16 of 34 boards were empty in one run; only a missing table counts as a layout change |
| 2025 scrapes have no landing times at Mehrabad | status `نشست \| پایان دریافت بار` with no time | Inbound delay features are empty for the 2025 data |
| The old scraper stored some dates a day off | `پروازکرد21:45` stored on the next day | The cleaner places event times on the day closest to the schedule and ignores the stored date |

## How the scraper copes

The ingest DAG (`dags/ingest_dag.py`) runs every 3 hours, because each board shows about a day and statuses change until the flight leaves.

| Concern | What it does |
| --- | --- |
| Slow or failing site | One `requests.Session` with a (5 s connect, 20 s read) timeout and 3 retries with exponential backoff on connection errors and 429/5xx, honouring `Retry-After` |
| Politeness | One airport at a time, 1.5 s apart, with a User-Agent that names the project and links to it |
| Fetching vs. parsing | Every page is saved gzipped to `data/raw_html/<date>/<time>/<airport id>.html.gz` with a `manifest.json` (14 days kept, about 1 MB a run). `scraper.reparse(<folder>)` parses a past scrape again after a parser fix |
| A page that can't be fetched, or whose tables changed | Recorded per airport (`fetch_failed`, `layout_changed`) instead of becoming an empty list. The other airports still load; the `check_pages` task then fails the run and `ScrapePageFailed` fires |
| Bad batches | Before loading: the data contract (pandera: non-empty scheduled time and flight number, known flight types, at most 5% of rows without airline, destination or status) and minimum row counts (200 in total, 50 at Mehrabad, 30 at Mashhad). A failure stops the load and fires `ScrapeValidationFailed` |
| Re-running | Rows are upserted on (flight number, scheduled time, airport). A row is only written when a value changed, and `updated_at` records when: loading the same scrape twice changes nothing (the second load of a run on 2026-11-06 reported 82 new or changed flights and 653 unchanged) |
| Persian spelling | Arabic and Persian `ي`/`ی` and `ك`/`ک`, zero-width non-joiners and spaces are unified in the cleaner and in every key built from names |
| Quality over time | Each run pushes rows per airport, failed pages, empty-cell shares (airline, status, registration, aircraft) and the share with an actual time to Prometheus; Grafana plots them |
