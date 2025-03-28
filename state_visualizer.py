from flask import Flask, render_template, request
import geopandas as gpd
import folium
import os

app = Flask(__name__)

# Load State Boundaries
state_boundaries = gpd.read_file("boundaries/India_State_Boundary_New.shp")

# Load Stations Data
stations = gpd.read_file("stations_cleaned.shp")

@app.route('/')
def index():
    states = sorted(state_boundaries["State_Name"].unique())
    return render_template('index.html', states=states)

@app.route('/map')
def generate_map():
    state_name = request.args.get("state")
    if not state_name:
        return "No state selected", 400

    state_boundary = state_boundaries[state_boundaries["State_Name"].str.lower() == state_name.lower()]
    if state_boundary.empty:
        return f"State '{state_name}' not found.", 404

    # Filter Stations in the Selected State
    stations_in_state = stations.sjoin(state_boundary, predicate="within")

    # Create Map
    state_center = state_boundary.geometry.centroid.iloc[0]
    m = folium.Map(location=[state_center.y, state_center.x], zoom_start=7)
    folium.GeoJson(state_boundary, name=f"{state_name} Boundary").add_to(m)

    for _, row in stations_in_state.iterrows():
        folium.Marker(
            location=[row.geometry.y, row.geometry.x],
            popup=f"Station: {row['stn_lbl']}",
            icon=folium.Icon(color="blue", icon="cloud"),
        ).add_to(m)
    
    # Save Map
    map_file = f"static/{state_name}_stations_map.html"
    m.save(map_file)
    return render_template('map.html', state=state_name, map_file=map_file)

if __name__ == '__main__':
    os.makedirs("static", exist_ok=True)
    app.run(debug=True)
