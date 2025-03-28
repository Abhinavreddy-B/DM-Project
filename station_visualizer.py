import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
import folium

# Load CSV
csv_file = "stations_with_location.csv"  # Change this to your actual file path
df = pd.read_csv(csv_file)

df.rename(columns={
    "state-label": "state_lbl",
    "state-value": "state_val",
    "city-label": "city_lbl",
    "city-value": "city_val",
    "station-label": "stn_lbl",
    "station-value": "stn_val"
}, inplace=True)

india_boundary = gpd.read_file("boundaries/India_Country_Boundary.shp")
india_boundary = india_boundary.to_crs(epsg=4326)

# Convert lat/lon to geometry (Point)
geometry = [Point(xy) for xy in zip(df["long"], df["lat"])]

# Create a GeoDataFrame
gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

gdf = gdf.sjoin(india_boundary, predicate="within")

gdf.to_csv("stations_cleaned.csv", index=False)
gdf.to_file("stations_cleaned.shp")