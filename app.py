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
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler


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

def interpolate_aqi(stations, aqi_data):
    """Perform spatial interpolation using IDW."""
    
    print(stations.head(10))
    # Merge AQI data with station locations
    stations = stations.merge(aqi_data, left_on='stn_val', right_on='station_id', how='inner')
    
    # Extract coordinates and AQI values
    points = stations[['long', 'lat']].values
    values = stations['aqi'].values
    
    # Generate grid points for interpolation
    grid_x, grid_y = np.meshgrid(
        np.linspace(points[:, 0].min(), points[:, 0].max(), 100),
        np.linspace(points[:, 1].min(), points[:, 1].max(), 100)
    )
    
    # Perform interpolation
    grid_z = griddata(points, values, (grid_x, grid_y), method='cubic')
    
    return grid_x, grid_y, grid_z

@app.route('/map')
def generate_map():
    state_name = request.args.get("state")
    # date = request.args.get("date")  # YYYY-MM-DD
    date = '2022-09-20'
    if not state_name or not date:
        return "State and date must be provided", 400
    
    year = date[:4]  # Extract year from date
    date_formatted = date[8:10] + "_" + date[5:7]  # Convert YYYY-MM-DD -> DD_MM
    
    # Load AQI data
    aqi_data = load_aqi_data_by_date(year, date_formatted)
    print(aqi_data)
    # Load state boundary and station metadata
    state_boundary = state_boundaries[state_boundaries["State_Name"].str.lower() == state_name.lower()]
    if state_boundary.empty:
        return f"State '{state_name}' not found.", 404
    
    stations_in_state = stations.sjoin(state_boundary, predicate="within")
    
    # Interpolate AQI
    grid_x, grid_y, grid_z = interpolate_aqi(stations_in_state, aqi_data)
    
    # Create Map
    state_center = state_boundary.geometry.centroid.iloc[0]
    m = folium.Map(location=[state_center.y, state_center.x], zoom_start=7)
    folium.GeoJson(state_boundary, name=f"{state_name} Boundary").add_to(m)
    
    # Add heatmap layer
    heat_data = [[grid_y[i, j], grid_x[i, j], grid_z[i, j]] for i in range(100) for j in range(100) if not np.isnan(grid_z[i, j])]
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


def predict_aqi_arima(station_id, periods=7):
    data = load_aqi_data(station_id)  

    if isinstance(data, list):
        df = pd.DataFrame(data)
    else:
        df = data

    if 'timestamp' not in df.columns or 'value' not in df.columns:
        return {'error': 'Invalid data format, missing "timestamp" or "value" column'}

    df['ds'] = pd.to_datetime(df['timestamp'])
    df.set_index('ds', inplace=True)

    model = ARIMA(df['value'], order=(5,1,0))
    model_fit = model.fit()

    forecast = model_fit.forecast(steps=periods).tolist()  # Convert to list

    future_dates = pd.date_range(start=df.index[-1], periods=periods+1, freq='D')[1:]

    predictions = [{'timestamp': str(future_dates[i]), 'predicted_aqi': forecast[i]} for i in range(periods)]

    return {'station_id': station_id, 'predictions': predictions}


def predict_aqi_lstm(station_id, periods=7):
    
    data = load_aqi_data(station_id)

    if isinstance(data, list):
        df = pd.DataFrame(data)
    else:
        df = data

    df['ds'] = pd.to_datetime(df['timestamp'])
    df.set_index('ds', inplace=True)

    scaler = MinMaxScaler()
    df['scaled_value'] = scaler.fit_transform(df[['value']])


    sequence_length = 30  # Use last 10 days to predict the futur
    X, y = [], []
    for i in range(len(df) - sequence_length):
        X.append(df['scaled_value'].values[i:i+sequence_length])
        y.append(df['scaled_value'].values[i+sequence_length])
    X, y = np.array(X), np.array(y)
    X = np.expand_dims(X, axis=-1)

    # Define LSTM model
    model = Sequential([
        LSTM(50, activation='relu', return_sequences=True, input_shape=(sequence_length, 1)),
        LSTM(50, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')

    # Train the model
    model.fit(X, y, epochs=50, batch_size=16, verbose=0)

    # Generate predictions
    last_sequence = df['scaled_value'].values[-sequence_length:].reshape(1, sequence_length, 1)
    predictions = []
    for _ in range(periods):
        pred_scaled = model.predict(last_sequence)[0, 0]
        pred_original = scaler.inverse_transform([[pred_scaled]])[0, 0]
        predictions.append(pred_original)
        last_sequence = np.roll(last_sequence, -1, axis=1)
        last_sequence[0, -1, 0] = pred_scaled 

    
    future_dates = pd.date_range(start=df.index[-1], periods=periods+1, freq='D')[1:]
    predictions = [{'timestamp': str(future_dates[i]), 'predicted_aqi': predictions[i]} for i in range(periods)]

    return {'station_id': station_id, 'predictions': predictions}




@app.route('/predict')
def predict_aqi():
    station_id = request.args.get('station_id')
    model=request.args.get('model')
    print(model,"saikiran")
    periods = int(request.args.get('periods', 10))  # Default to 7 days
    predictions=None
    if(model=='arima'):
        predictions = predict_aqi_arima(station_id, periods)
    else:
        predictions = predict_aqi_lstm(station_id, periods)
        
        

    return jsonify({
        'station_id': station_id,
        'predictions': predictions['predictions'],
    })



if __name__ == '__main__':
    os.makedirs("static", exist_ok=True)
    app.run(debug=True)
