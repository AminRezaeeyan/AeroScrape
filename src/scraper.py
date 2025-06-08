import requests
from bs4 import BeautifulSoup
import pandas as pd
from urllib.parse import urljoin
from config import Config

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
}

def scrape_table(soup, table_id, airport_name):
    table_container = soup.find('div', id=table_id)
    if not table_container:
        return []

    table = table_container.find('table')
    if not table:
        return []

    rows = table.find_all('tr')
    data = []
    for row in rows[1:]:  # Skip header
        cols = row.find_all('td')
        if len(cols) >= 9:
            flight_data = {
                'scheduled_time': cols[0].text.strip(),
                'airline': cols[1].text.strip(),
                'flight_number': cols[2].text.strip(),
                'destination_or_origin': cols[3].text.strip(),
                'status': cols[4].text.strip(),
                'counter': cols[5].text.strip(),
                'actual_time': cols[6].text.strip(),
                'register': cols[7].text.strip(),
                'aircraft': cols[8].text.strip(),
                'airport': airport_name,
                'flight_type': Config.FLIGHT_TABLE_IDS.get(table_id, 'Unknown')
            }
            data.append(flight_data)
    return data

def scrape_airport_data(airport_name, airport_url):
    full_url = urljoin(Config.BASE_URL, airport_url)
    try:
        response = requests.get(full_url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        all_flights = []
        for table_id in Config.FLIGHT_TABLE_IDS:
            flights = scrape_table(soup, table_id, airport_name)
            all_flights.extend(flights)
        return all_flights
    except requests.RequestException as e:
        print(f"Error fetching data for {airport_name}: {e}")
        return []

def get_all_flights_data() -> pd.DataFrame:
    all_data = []
    for airport in Config.AIRPORTS:
        print(f"Scraping data for {airport['name']}...")
        airport_data = scrape_airport_data(airport['name'], airport['url'])
        all_data.extend(airport_data)

    df = pd.DataFrame(all_data)

    if not df.empty:
        # Split destination_or_origin into destination and origin based on flight_type
        df['destination'] = df.apply(
            lambda row: row['destination_or_origin'] if 'departure' in row['flight_type'].lower() else None,
            axis=1
        )
        df['origin'] = df.apply(
            lambda row: row['destination_or_origin'] if 'arrival' in row['flight_type'].lower() else None,
            axis=1
        )

    return df