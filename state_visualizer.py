import geopandas as gpd
import folium

# Load State Boundaries
state_boundaries = gpd.read_file("boundaries/India_State_Boundary_New.shp")

print(state_boundaries)
# Load Stations Data
stations = gpd.read_file("stations_cleaned.shp")  # Filtered stations file

# Get State Name from User
state_name = input("Enter the state name: ").strip()

# Filter State Boundary
state_boundary = state_boundaries[state_boundaries["State_Name"].str.lower() == state_name.lower()]

if state_boundary.empty:
    print(f"State '{state_name}' not found in boundaries.")
    exit()

# Filter Stations in the Selected State (Spatial Join)
stations_in_state = stations.sjoin(state_boundary, predicate="within")

# Create Map Centered on State
state_center = state_boundary.geometry.centroid.iloc[0]
m = folium.Map(location=[state_center.y, state_center.x], zoom_start=7)

# Add State Boundary to Map
folium.GeoJson(state_boundary, name=f"{state_name} Boundary").add_to(m)

# Add Station Markers
for _, row in stations_in_state.iterrows():
    folium.Marker(
        location=[row.geometry.y, row.geometry.x],
        popup=f"Station: {row['stn_lbl']}",
        icon=folium.Icon(color="blue", icon="cloud"),
    ).add_to(m)

# Save and Show Map
map_file = f"{state_name}_stations_map.html"
m.save(map_file)
print(f"Map saved as '{map_file}'. Open this file in a browser.")
