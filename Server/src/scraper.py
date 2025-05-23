import requests
from bs4 import BeautifulSoup
import pandas as pd
from urllib.parse import urljoin

base_url = "https://fids.airport.ir/"

# List of airports with name and URL
airports = [
    {"name": "فرودگاه مهرآباد", "url": "/2/اطلاعات-پرواز-فرودگاه-مهرآباد"},
    {"name": "فرودگاه مشهد", "url": "/102/اطلاعات-پرواز-فرودگاه-مشهد"},
    {"name": "فرودگاه شيراز", "url": "/1/اطلاعات-پرواز-فرودگاه-شيراز"},
    {"name": "فرودگاه تبريز", "url": "/103/اطلاعات-پرواز-فرودگاه-تبريز"},
    {"name": "فرودگاه اصفهان", "url": "/114/اطلاعات-پرواز-فرودگاه-اصفهان"},
    {"name": "فرودگاه اهواز", "url": "/401/اطلاعات-پرواز-فرودگاه-اهواز"},
    {"name": "فرودگاه بوشهر", "url": "/104/اطلاعات-پرواز-فرودگاه-بوشهر"},
    {"name": "فرودگاه بندرعباس", "url": "/117/اطلاعات-پرواز-فرودگاه-بندرعباس"},
    {"name": "فرودگاه کرمان", "url": "/201/اطلاعات-پرواز-فرودگاه-کرمان"},
    {"name": "فرودگاه ساري", "url": "/106/اطلاعات-پرواز-فرودگاه-ساري"},
    {"name": "فرودگاه يزد", "url": "/107/اطلاعات-پرواز-فرودگاه-يزد"},
    {"name": "فرودگاه اروميه", "url": "/110/اطلاعات-پرواز-فرودگاه-اروميه"},
    {"name": "فرودگاه کرمانشاه", "url": "/111/اطلاعات-پرواز-فرودگاه-کرمانشاه"},
    {"name": "فرودگاه رشت", "url": "/203/اطلاعات-پرواز-فرودگاه-رشت"},
    {"name": "فرودگاه زاهدان", "url": "/109/اطلاعات-پرواز-فرودگاه-زاهدان"},
    {"name": "فرودگاه آبادان", "url": "/301/اطلاعات-پرواز-فرودگاه-آبادان"},
    {"name": "فرودگاه گرگان", "url": "/202/اطلاعات-پرواز-فرودگاه-گرگان"},
    {"name": "فرودگاه همدان", "url": "/112/اطلاعات-پرواز-فرودگاه-همدان"},
    {"name": "فرودگاه اردبيل", "url": "/113/اطلاعات-پرواز-فرودگاه-اردبيل"},
    {"name": "فرودگاه ايلام", "url": "/105/اطلاعات-پرواز-فرودگاه-ايلام"},
    {"name": "فرودگاه بيرجند", "url": "/204/اطلاعات-پرواز-فرودگاه-بيرجند"},
    {"name": "فرودگاه سنندج", "url": "/402/اطلاعات-پرواز-فرودگاه-سنندج"},
    {"name": "فرودگاه شهرکرد", "url": "/108/اطلاعات-پرواز-فرودگاه-شهرکرد"},
    {"name": "فرودگاه بجنورد", "url": "/901/اطلاعات-پرواز-فرودگاه-بجنورد"},
    {"name": "فرودگاه لارستان", "url": "/601/اطلاعات-پرواز-فرودگاه-لارستان"},
    {"name": "فرودگاه خرم آباد", "url": "/701/اطلاعات-پرواز-فرودگاه-خرم-آباد"},
    {"name": "فرودگاه پارس آبادمغان", "url": "/702/اطلاعات-پرواز-فرودگاه-پارس-آبادمغان"},
    {"name": "فرودگاه سمنان", "url": "/801/اطلاعات-پرواز-فرودگاه-سمنان"},
    {"name": "فرودگاه شاهرود", "url": "/802/اطلاعات-پرواز-فرودگاه-شاهرود"},
    {"name": "فرودگاه نوشهر", "url": "/1201/اطلاعات-پرواز-فرودگاه-نوشهر"},
    {"name": "فرودگاه ياسوج", "url": "/1001/اطلاعات-پرواز-فرودگاه-ياسوج"},
    {"name": "فرودگاه زنجان", "url": "/501/اطلاعات-پرواز-فرودگاه-زنجان"},
    {"name": "فرودگاه اراک", "url": "/1401/اطلاعات-پرواز-فرودگاه-اراک"},
    {"name": "فرودگاه زابل", "url": "/1501/اطلاعات-پرواز-فرودگاه-زابل"}
]

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

# Map of table IDs to flight types
flight_table_ids = {
    "input": "Domestic Arrival",
    "output": "Domestic Departure",
    "internal": "International Arrival",
    "external": "International Departure"
}

def scrape_table(soup, table_id, airport_name):
    table_container = soup.find('div', id=table_id)
    if not table_container:
        print(f"No section '{table_id}' for {airport_name}")
        return []

    table = table_container.find('table')
    if not table:
        print(f"No table found in '{table_id}' for {airport_name}")
        return []

    rows = table.find_all('tr')
    data = []
    for row in rows[1:]:  # Skip header
        cols = row.find_all('td')
        if len(cols) >= 9:
            flight_data = {
                'Scheduled Time': cols[0].text.strip(),
                'Airline': cols[1].text.strip(),
                'Flight Number': cols[2].text.strip(),
                'Destination': cols[3].text.strip(),
                'Status': cols[4].text.strip(),
                'Counter': cols[5].text.strip(),
                'Actual Time': cols[6].text.strip(),
                'Register': cols[7].text.strip(),
                'Aircraft': cols[8].text.strip(),
                'Airport': airport_name,
                'Flight Type': flight_table_ids.get(table_id, 'Unknown')
            }
            data.append(flight_data)
    return data

def scrape_airport_data(airport_name, airport_url):
    full_url = urljoin(base_url, airport_url)
    try:
        response = requests.get(full_url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        all_flights = []
        for table_id in flight_table_ids.keys():
            flights = scrape_table(soup, table_id, airport_name)
            all_flights.extend(flights)
        
        return all_flights
    except requests.RequestException as e:
        print(f"Error fetching data for {airport_name}: {e}")
        return []

def get_all_flights_data() -> pd.DataFrame:
    """
    Scrapes flight data from all Iranian airports and returns it as a pandas DataFrame.
    
    Returns:
        pd.DataFrame: A DataFrame containing all flight information with columns:
            - Scheduled Time
            - Airline
            - Flight Number
            - Destination
            - Status
            - Counter
            - Actual Time
            - Register
            - Aircraft
            - Airport
            - Flight Type
    """
    all_data = []
    for airport in airports:
        print(f"Scraping data for {airport['name']}...")
        airport_data = scrape_airport_data(airport['name'], airport['url'])
        all_data.extend(airport_data)
    
    return pd.DataFrame(all_data)