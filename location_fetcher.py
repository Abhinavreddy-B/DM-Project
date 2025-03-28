import requests
from urllib.parse import quote
import pandas as pd
import os
from tqdm import tqdm

tqdm.pandas()

CHECKPOINT_FILE = "stations_with_location.csv"

def get_lat_lon_from_query(query):
    encoded_query = quote(query)  # URL encode the query
    url = f"https://nominatim.openstreetmap.org/search?q={encoded_query}&format=json"

    headers = {
        "User-Agent": "YourAppName/1.0 (@gmail.com)"  # Set your app name & email
    }
    
    response = requests.get(url, headers=headers).json()
    
    if response:
        return float(response[0]["lat"]), float(response[0]["lon"])
    
    return None

def get_lat_lon(station_name, city, state):
    station_query = f"{station_name}"
    city_query = f"{city}"
    state_query = f"{state}"
    
    station_result = get_lat_lon_from_query(station_query)
    if station_result:
        return station_result
    
    city_result = get_lat_lon_from_query(city_query)
    if city_result:
        return city_result
    
    state_result = get_lat_lon_from_query(state_query)
    if state_result:
        return state_result

    return None

# Load existing data if checkpoint file exists
if os.path.exists(CHECKPOINT_FILE):
    df = pd.read_csv(CHECKPOINT_FILE)
else:
    df = pd.read_csv("stations.csv")
    df[['lat', 'long']] = None  # Initialize empty columns

# Process only rows with missing coordinates
for index, row in tqdm(df.iterrows(), total=len(df)):
    if pd.isna(row.get('lat')) or pd.isna(row.get('long')):
        lat_lon = get_lat_lon(row['station-label'].split('-')[0].strip(), row['city-label'], row['state-label'])
        if lat_lon:
            df.at[index, 'lat'], df.at[index, 'long'] = lat_lon

    # Save progress every 10 rows
    if index % 10 == 0:
        df.to_csv(CHECKPOINT_FILE, index=False)

# Final save
df.to_csv(CHECKPOINT_FILE, index=False)
print("Processing complete. Data saved.")
