import requests
import base64
import json
import csv
import pandas as pd
from tabulate import tabulate
import urllib
import os
from tqdm import tqdm

url = "https://airquality.cpcb.gov.in/dataRepository/all_india_stationlist"
data = {"e30": ""}

response = requests.post(url, data=data)
encoded_data = response.text

decoded_bytes = base64.b64decode(encoded_data)
decoded_json = json.loads(decoded_bytes.decode("utf-8"))

states = decoded_json['dropdown']['states']
cities_for_state = decoded_json['dropdown']['cities']
stations_for_city = decoded_json['dropdown']['stations']

stations_list = []

for state in states:
    for city in cities_for_state[state['value']]:
        for station in stations_for_city[city['value']]:
            stations_list.append([
                state['label'],
                state['value'],
                city['label'],
                city['value'],
                station['label'],
                station['value'],
            ])

df = pd.DataFrame(stations_list, columns=['state-label', 'state-value', 'city-label', 'city-value', 'station-label', 'station-value'])
df.to_csv('stations.csv', index=False, header=True)
print(tabulate(df, tablefmt='psql', headers=df.columns))

summary = [
    ["No of states", df['state-label'].nunique()],
    ["No of cities", df['city-label'].nunique()],
    ["No of stations", df['station-label'].nunique()]
]

print(tabulate(summary, headers=["Summary", "Count"], tablefmt="psql"))

url = "https://airquality.cpcb.gov.in/dataRepository/file_Path"  # Replace with actual endpoint

pbar = tqdm(df.iterrows(), total=len(df), desc='Downloading Data')
for _, row in pbar:
    pbar.set_postfix(state=row["state-label"])
    data = {
        "station_id": row["station-value"],
        "station_name": row["station-label"],
        "state": row["state-label"],
        "city": row["city-label"],
        "year": "",
        "frequency": "daily",
        "dataType": "stationLevel"
    }

    json_str = json.dumps(data, ensure_ascii=False, separators=(',', ':'))  # Ensure proper encoding
    encoded_key = base64.b64encode(json_str.encode()).decode()  # Encode properly

    response = requests.post(url, data=encoded_key)
    encoded_data = response.text

    decoded_bytes = base64.b64decode(encoded_data)
    decoded_json = json.loads(decoded_bytes.decode("utf-8"))

    data = decoded_json['data']

    year_pbar = tqdm(data, desc=row['station-label'], leave=False)
    for item in year_pbar:
        year_pbar.set_postfix(year=item['year'])
        filepath, year = item['filepath'], item['year']

        base_url = "https://airquality.cpcb.gov.in/dataRepository/download_file?file_name="

        response = requests.get(base_url + filepath, stream=True)
        if response.status_code == 200:
            folder = f"raw_data/station_level/{row['state-label']}_{row['city-label']}"  
            os.makedirs(folder, exist_ok=True)  

            file_path = os.path.join(folder, f"{row['station-label']}_{row['station-value']}_{year}.xlsx")  

            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
        else:
            print("Download failed:", response.status_code)