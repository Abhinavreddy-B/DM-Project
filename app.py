from flask import Flask, render_template, request, jsonify, send_file
import geopandas as gpd
import folium
from folium.plugins import HeatMap
import os
import random
import json
import pandas as pd
import matplotlib.pyplot as plt
import io   
import base64
import numpy as np
from scipy.interpolate import griddata
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from scipy.interpolate import Rbf


app = Flask(__name__)

# Load State Boundaries
state_boundaries = gpd.read_file("boundaries/India_State_Boundary_New.shp")

# Load Stations Data
stations = gpd.read_file("stations_cleaned.shp")

@app.route('/')
def index():
    states = sorted(state_boundaries["State_Name"].unique())
    return render_template('index.html', states=states)

def load_aqi_data_by_date(year, date):
    """Load AQI data for the given year and date."""
    file_path = f"database/{year}.parquet"
    df = pd.read_parquet(file_path)
    df = df[df['date'] == date]  # Filter for the given date
    return df

def interpolate_aqi(stations, aqi_data, state_boundary):
    """
    Interpolate AQI over a state's area using all stations.

    Parameters:
    - stations: DataFrame with columns ['stn_val', 'lat', 'long']
    - aqi_data: DataFrame with columns ['station_id', 'aqi']
    - state_boundary: GeoDataFrame with a single polygon geometry for the state

    Returns:
    - grid_x, grid_y: meshgrid arrays of lon/lat
    - grid_z_masked: interpolated AQI values masked to the state
    - mask: boolean array showing which points are inside the state
    """
    
    # Merge AQI data with station locations
    merged = stations.merge(aqi_data, left_on='stn_val', right_on='station_id', how='inner')
    
    # Coordinates and AQI values from all stations
    points = merged[['long', 'lat']].values
    values = merged['aqi'].values

    # Get the state's polygon geometry
    state_geom = state_boundary.geometry.unary_union

    # Create a grid over the state's bounding box
    minx, miny, maxx, maxy = state_geom.bounds
    grid_x, grid_y = np.meshgrid(
        np.linspace(minx, maxx, 100),
        np.linspace(miny, maxy, 100)
    )

    # Interpolate AQI over entire grid using all stations
    grid_z = griddata(points, values, (grid_x, grid_y), method='cubic')

    # Create a GeoDataFrame of grid points
    grid_points_flat = pd.DataFrame({
        'lon': grid_x.ravel(),
        'lat': grid_y.ravel()
    })
    grid_points_gdf = gpd.GeoDataFrame(
        grid_points_flat,
        geometry=gpd.points_from_xy(grid_points_flat['lon'], grid_points_flat['lat']),
        crs='EPSG:4326'
    )

    # Create mask of which points are inside the state polygon
    inside_mask = grid_points_gdf.within(state_geom).values
    mask = inside_mask.reshape(grid_x.shape)

    # Mask the interpolated result
    grid_z_masked = np.where(mask, grid_z, np.nan)

    return grid_x, grid_y, grid_z_masked, mask

@app.route('/map')
def generate_map():
    state_name = request.args.get("state")
    date = request.args.get("date", None)

    if not state_name:
        return "State must be provided", 400
    
        
    # print(aqi_data)
    # Load state boundary and station metadata
    state_boundary = state_boundaries[state_boundaries["State_Name"].str.lower() == state_name.lower()]
    
    if state_boundary.empty:
        return f"State '{state_name}' not found.", 404
    
    stations_in_state = stations.sjoin(state_boundary, predicate="within")
            
    # Create Map
    state_center = state_boundary.geometry.centroid.iloc[0]
    m = folium.Map(location=[state_center.y, state_center.x], zoom_start=7)
    folium.GeoJson(state_boundary, name=f"{state_name} Boundary").add_to(m)

    if date is not None:
        year = date[:4]  # Extract year from date
        date_formatted = date[8:10] + "_" + date[5:7]  # Convert YYYY-MM-DD -> DD_MM

        aqi_data = load_aqi_data_by_date(year, date_formatted)
        aqi_filtered_data = aqi_data[aqi_data["station_id"].isin(stations_in_state["stn_val"])]
        no_of_available_data_points = len(aqi_filtered_data)
        
        if no_of_available_data_points < 3:
            return f"Less than 3 data points available for {state_name} on {date}", 404
        
        # Interpolate AQI
        grid_x, grid_y, grid_z_masked, mask = interpolate_aqi(stations, aqi_data, state_boundary)

        # Add heatmap layer
        heat_data = [
            [grid_y[i, j], grid_x[i, j], grid_z_masked[i, j]]
            for i in range(grid_x.shape[0])
            for j in range(grid_x.shape[1])
            if not np.isnan(grid_z_masked[i, j])
        ]
        HeatMap(heat_data, radius=15).add_to(m)
    
    # Add stations as markers
    for _, row in stations_in_state.iterrows():
        station_label = row['stn_lbl']
        station_id = row['stn_val']
        lat, lon = row.geometry.y, row.geometry.x
        
        popup_html = f"""
        <b>{station_label}</b><br>
        <button onclick="window.parent.postMessage({repr({'station': station_id, 'action': 'timeseries'})}, '*')">
            Show Time Series
        </button>
        <button onclick="window.parent.postMessage({repr({'station': station_id, 'action': 'predict'})}, '*')">
            Predict AQI
        </button>
        """
        folium.Marker(
            location=[lat, lon],
            popup=popup_html,
            icon=folium.Icon(color="blue", icon="cloud"),
        ).add_to(m)
    
    # Save Map
    map_file = f"static/{state_name}_{date}_stations_map.html"
    m.save(map_file)
    
    return render_template('map.html', state=state_name, date=date, map_file=map_file)

DATABASE_FOLDER = "database"

def load_aqi_data(station_id):
    all_years = [f.split('.')[0] for f in os.listdir(DATABASE_FOLDER) if f.endswith(".parquet")]
    all_dfs = []
    
    for year in all_years:
        file_path = os.path.join(DATABASE_FOLDER, f"{year}.parquet")
        if not os.path.exists(file_path):
            continue
        
        df = pd.read_parquet(file_path)
        df_filtered = df[df["station_id"] == station_id]
        
        if df_filtered.empty:
            continue
        
        df_filtered["formatted_date"] = df_filtered["date"].apply(lambda x: f"{year}-{x[3:]}-{x[:2]}")
        all_dfs.append(df_filtered[["formatted_date", "aqi"]].rename(columns={"formatted_date": "timestamp", "aqi": "value"}))
    
    if not all_dfs:
        return None
    
    final_df = pd.concat(all_dfs).sort_values(by="timestamp", ascending=True)
    return final_df.to_dict(orient="records")

@app.route('/timeseries')
def get_timeseries():
    station_id = request.args.get("station_id")
    
    if not station_id:
        return jsonify({"error": "No station ID provided"}), 400
    
    time_series_data = load_aqi_data(station_id)
    
    if time_series_data is None:
        return jsonify({"error": "No data found for the given station ID"}), 404
    
    return jsonify({"station_id": station_id, "data": time_series_data})


def predict_aqi_arima(station_id, periods=6):
    data = load_aqi_data(station_id)  # Load daily data

    if isinstance(data, list):
        df = pd.DataFrame(data)
    else:
        df = data

    if 'timestamp' not in df.columns or 'value' not in df.columns:
        return {'error': 'Invalid data format, missing "timestamp" or "value" column'}

    df['ds'] = pd.to_datetime(df['timestamp'])
    df.set_index('ds', inplace=True)

    # Resample to monthly average
    monthly_df = df['value'].resample('M').mean().dropna()

    if len(monthly_df) < 13:
        return {'error': 'Not enough data to train a monthly model'}

    model = ARIMA(monthly_df, order=(2, 1, 2))  # You can tune these params
    model_fit = model.fit()

    forecast = model_fit.forecast(steps=periods).tolist()

    future_dates = pd.date_range(start=monthly_df.index[-1] + pd.offsets.MonthEnd(1),
                                 periods=periods, freq='M')

    predictions = [
        {'timestamp': str(future_dates[i].date()), 'predicted_aqi': forecast[i]}
        for i in range(periods)
    ]
    return {'station_id': station_id, 'predictions': predictions}


class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Take output of the last time step
        return out

def predict_aqi_lstm(station_id, periods=6):
    data = load_aqi_data(station_id)

    if isinstance(data, list):
        df = pd.DataFrame(data)
    else:
        df = data

    if 'timestamp' not in df.columns or 'value' not in df.columns:
        return {'error': 'Invalid data format, missing "timestamp" or "value" column'}

    df['ds'] = pd.to_datetime(df['timestamp'])
    df.set_index('ds', inplace=True)

    # Resample to monthly averages
    monthly_df = df['value'].resample('M').mean().dropna()

    if len(monthly_df) < 13:
        return {'error': 'Not enough monthly data to train the model'}

    # Normalize
    scaler = MinMaxScaler()
    scaled_values = scaler.fit_transform(monthly_df.values.reshape(-1, 1)).flatten()

    sequence_length = 12  # Use last 12 months to predict next
    X, y = [], []
    for i in range(len(scaled_values) - sequence_length):
        X.append(scaled_values[i:i+sequence_length])
        y.append(scaled_values[i+sequence_length])
    X, y = np.array(X), np.array(y)
    X = np.expand_dims(X, axis=-1)

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = LSTMModel()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train model
    for epoch in range(50):
        for xb, yb in loader:
            out = model(xb)
            loss = criterion(out, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Predict future
    model.eval()
    with torch.no_grad():
        last_sequence = torch.tensor(scaled_values[-sequence_length:], dtype=torch.float32).view(1, sequence_length, 1)
        predictions = []
        for _ in range(periods):
            pred_scaled = model(last_sequence).item()
            pred_original = scaler.inverse_transform([[pred_scaled]])[0, 0]
            predictions.append(pred_original)

            last_sequence = torch.roll(last_sequence, -1, dims=1)
            last_sequence[0, -1, 0] = pred_scaled

    # Generate monthly future dates
    last_date = monthly_df.index[-1] + pd.offsets.MonthEnd(1)
    future_dates = pd.date_range(start=last_date, periods=periods, freq='M')

    predictions = [{'timestamp': str(future_dates[i].date()), 'predicted_aqi': predictions[i]} for i in range(periods)]

    return {'station_id': station_id, 'predictions': predictions}





@app.route('/predict')
def predict_aqi():
    station_id = request.args.get('station_id')
    model=request.args.get('model')
    print(model,"saikiran")
    periods = int(request.args.get('periods', 30))  # Default to 7 days
    predictions=None
    if(model=='arima'):
        predictions = predict_aqi_arima(station_id, periods)
    else:
        predictions = predict_aqi_lstm(station_id, periods)
        
        
    if 'error' in predictions:
        return jsonify(predictions), 400

    return jsonify({
        'station_id': station_id,
        'predictions': predictions['predictions'],
    })



if __name__ == '__main__':
    os.makedirs("static", exist_ok=True)
    app.run(debug=True)
