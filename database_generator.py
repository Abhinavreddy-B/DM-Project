import os
import pandas as pd
from tqdm import tqdm
import numpy as np


def get_years(directory):
    possible_years = set()
    for root, _, files in os.walk(directory):
        for file in files:
            possible_years.add(file.strip('.xlsx').split("_")[-1])
    
    return sorted(list(possible_years))

# years = get_years("raw_data/station_level")
years = [2023]
months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
month_map = {
    'January': '01',
    'February': '02',
    'March': '03',
    'April': '04',
    'May': '05',
    'June': '06',
    'July': '07',
    'August': '08',
    'September': '09',
    'October': '10',
    'November': '11',
    'December': '12',
}

# def convert_to_parquet(root_folder, output_folder):
#     os.makedirs(output_folder, exist_ok=True)
    
#     station_metadata = []
#     yearly_data = {}
    
#     for city in tqdm(os.listdir(root_folder), desc="Processing Cities"):
#         city_path = os.path.join(root_folder, city)
#         if not os.path.isdir(city_path):
#             continue
        
#         for file in os.listdir(city_path):
#             if file.endswith(".csv"):  # Ensure it's a CSV file
#                 parts = file.replace(".csv", "").split("_")
#                 if len(parts) < 2:
#                     continue  # Skip malformed file names
                
#                 year = parts[0]
#                 station_name = "_".join(parts[1:])
                
#                 file_path = os.path.join(city_path, file)
#                 df = pd.read_csv(file_path, index_col=0)  # First column is day (index)
                
#                 df.index = df.index.astype(int)  # Ensure days are integers
#                 df = df.stack().reset_index()
#                 df.columns = ["day", "month", "aqi"]
#                 df["month"] = df["month"].astype(int)
#                 df["date"] = pd.to_datetime(df[["month", "day"]].assign(year=int(year)))
#                 df.drop(columns=["month", "day"], inplace=True)
                
#                 station_id = hash(station_name) % (10**6)  # Generate a station ID
#                 df.insert(0, "station_id", station_id)
                
#                 if year not in yearly_data:
#                     yearly_data[year] = []
#                 yearly_data[year].append(df)
                
#                 # Collect station metadata
#                 station_metadata.append([station_id, station_name, city])
    
#     # Save yearly AQI data
#     for year, data_list in yearly_data.items():
#         yearly_df = pd.concat(data_list, ignore_index=True)
#         yearly_df.to_parquet(os.path.join(output_folder, f"aqi_{year}.parquet"), index=False)
    
#     # Save station metadata
#     station_df = pd.DataFrame(station_metadata, columns=["station_id", "name", "city"])
#     station_df.to_csv(os.path.join(output_folder, "stations.csv"), index=False)

# # Example usage
# convert_to_parquet("/path/to/city_folders", "/path/to/aqi_data")

df = pd.read_csv('stations_cleaned.csv')

for year in tqdm(years, desc="Processing Years"):
    year_df = []
    for index, row in tqdm(df.iterrows(), desc=f"Processing Year {year}", total=len(df)):
        state, city, station, station_id = row['state_lbl'], row['city_lbl'], row['stn_lbl'], row['stn_val']
    
        file_path = f'raw_data/station_level/{state}_{city}/{station}_{station_id}_{year}.xlsx'
        if not os.path.exists(file_path):
            continue
        
        station_year_df = pd.read_excel(file_path, header=None)
        # print(station_year_df.iloc[:, 0][2])
        for idx, value in enumerate(station_year_df.iloc[:, 0]):  # Check first column
            if str(value) in ['Day', 'Date']:
                start_row = idx
                break
        else:
            raise ValueError("No valid header found in the file.")

        # Read the Excel again with the detected header row
        station_year_df = pd.read_excel(file_path, skiprows=start_row+1,header=None, names=['Date', 'January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'])
        # print(file_path)
        
        print(file_path)
        for month in months:
            for date in range(1, 32):
                if date < 10:
                    date_str = f"0{date}_{month_map[month]}"
                else:
                    date_str = f"{date}_{month_map[month]}"
                
                val = station_year_df.loc[station_year_df['Date'].isin([date, f"0{date}"])][month].values[0]
                if pd.isna(val) or val == 'NA ':
                    continue
                else:
                    val = float(val)
                # print(val)
                # df.loc[df['stn_val'] == station_id, month] = station_year_df.loc[station_year_df['Date'] == date][month].values
                
                year_df.append([station_id, date_str, val])
        # print(df.head(10))
    
    year_df = pd.DataFrame(year_df, columns=['station_id', 'date', 'aqi'])
    year_df.to_parquet(f'database/{year}.parquet', index=False)
    print(year_df)

# print(df)