import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
from urllib.parse import urljoin
import uuid

base_url = "https://fids.airport.ir/"

# List of airports with their corresponding URLs from the select options
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

def scrape_departure_table(airport_name, airport_url):
    full_url = urljoin(base_url, airport_url)
    try:
        response = requests.get(full_url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find the departure table (tab with id="output")
        departure_table = soup.find('div', id='output')
        if not departure_table:
            print(f"No departure table found for {airport_name}")
            return []

        # Find the table within the departure tab
        table = departure_table.find('table')
        if not table:
            print(f"No table found in departure tab for {airport_name}")
            return []

        rows = table.find_all('tr')
        data = []
        for row in rows[1:]:  # Skip header row
            cols = row.find_all('td')
            if len(cols) >= 9:
                flight_data = {
                    'زمان برنامه ای': cols[0].text.strip(),
                    'ایرلاین': cols[1].text.strip(),
                    'شماره پرواز': cols[2].text.strip(),
                    'مقصد': cols[3].text.strip(),
                    'وضعیت': cols[4].text.strip(),
                    'کانتر': cols[5].text.strip(),
                    'تاریخ و زمان واقعی': cols[6].text.strip(),
                    'رجیستر': cols[7].text.strip(),
                    'هواپیما': cols[8].text.strip(),
                    'فرودگاه': airport_name
                }
                data.append(flight_data)
        return data
    except requests.RequestException as e:
        print(f"Error fetching data for {airport_name}: {e}")
        return []

def main():
    all_data = []
    for airport in airports:
        print(f"Scraping data for {airport['name']}...")
        airport_data = scrape_departure_table(airport['name'], airport['url'])
        all_data.extend(airport_data)
    
    if all_data:
        df = pd.DataFrame(all_data)
        df.to_csv('C:/Users/ASUS/Desktop/AUT/4th_Term/DB/Data/raw/flight_data.csv', index=False, encoding='utf-8-sig')
        print("Data saved to flight_departures.csv")
    else:
        print("No data was scraped.")

if __name__ == "__main__":
    main()