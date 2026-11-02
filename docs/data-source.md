# Data source: fids.airport.ir

AeroScrape reads the public flight information display (FIDS) pages of Iran Airports Company, one page per airport, listed in `config.yaml`. These notes record how the pages are structured and where they are unreliable. Last checked 2026-11-02 against the Mehrabad and Mashhad pages.

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
| The actual time cell can hold an estimate | CPN024: cell `07:10`, status `تخمین نشست 07:10 \| نشست07:15` | Not handled: the status text holds the real landing time. To fix with status parsing |
| Status text embeds times and varies in spelling | `پروازکرد04:57`, older data `پرواز كرد` | Stored as-is. Normalization is planned (upgrade plan, Phase 2) |
| The date column disagrees with the weekday for some international flights | Monday flight dated `2026-11-04`; one cell reads `m 2026-11-04` | Not used. Domestic rows' dates match their weekday |
| Airline cell can be a logo without text | `<img src=".../IRM.png">` | Falls back to the logo's airline code (`IRM`) |
| The page shows a rolling window | about one day of flights per airport | Scraping only once a day misses flights and final statuses |
