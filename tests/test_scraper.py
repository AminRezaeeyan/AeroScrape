from bs4 import BeautifulSoup

from aeroscrape.scraping.scraper import scrape_table

COLUMNS = ["08:00", "ماهان", "IRM1070", "زاهدان", "پرواز كرد", "3", "08:20", "EPMEI", "RJ85"]


def _page(rows: list[list[str]]) -> BeautifulSoup:
    header = "<tr><th>header</th></tr>"
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in rows)
    return BeautifulSoup(f'<div id="output"><table>{header}{body}</table></div>', "html.parser")


def test_scrape_table_maps_columns() -> None:
    rows = scrape_table(_page([COLUMNS]), "output", "فرودگاه مهرآباد")

    assert len(rows) == 1
    assert rows[0]["flight_number"] == "IRM1070"
    assert rows[0]["actual_time"] == "08:20"
    assert rows[0]["airport"] == "فرودگاه مهرآباد"
    assert rows[0]["flight_type"] == "Domestic Departure"


def test_scrape_table_returns_nothing_for_a_missing_table() -> None:
    assert scrape_table(_page([COLUMNS]), "input", "فرودگاه مهرآباد") == []
