import time
import folium
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
from folium.plugins import HeatMap
from streamlit_folium import st_folium
from geopy.geocoders import Nominatim

st.set_page_config(page_title="Food Truck Hotspot Finder", layout="wide")
st.title("🚚 Food Truck Location Optimizer")

st.sidebar.header("Configuration Settings")

uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=["csv"],max_upload_size=1024)
num_clusters = st.sidebar.slider("Number of Food Truck Hubs", min_value=1, max_value=10, value=3)
top_n = st.sidebar.slider("Top Locations to Analyze", min_value=5, max_value=50, value=20)

def clean_rate_series(series: pd.Series) -> pd.Series:
    cleaned = series.astype(str).str.split('/').str[0].str.strip()
    cleaned = cleaned.replace(['NEW', '-', 'nan', 'None'], np.nan)
    return pd.to_numeric(cleaned, errors='coerce')

@st.cache_data
def load_large_csv(file) -> pd.DataFrame:
    required_cols = {'rate', 'location', 'votes'}
    df_chunk = pd.read_csv(file, usecols=lambda c: c.lower() in required_cols)
    df_chunk.columns = df_chunk.columns.str.lower()
    return df_chunk

@st.cache_data
def geocode_locations(locations: tuple) -> tuple:
    geolocator = Nominatim(user_agent="food_truck_app_v2", timeout=5)
    lats, lons = [], []
    for loc in locations:
        try:
            loc_data = geolocator.geocode(f"{loc}, Bangalore, India")
            if loc_data:
                lats.append(loc_data.latitude)
                lons.append(loc_data.longitude)
            else:
                lats.append(None)
                lons.append(None)
            time.sleep(1)
        except Exception:
            lats.append(None)
            lons.append(None)
    return lats, lons

if uploaded_file is not None:
    try:
        with st.spinner("Loading dataset efficiently..."):
            df = load_large_csv(uploaded_file)

        df['cleaned_rate'] = clean_rate_series(df['rate'])
        df.dropna(subset=['location', 'cleaned_rate'], inplace=True)

        competition = df['location'].value_counts().reset_index()
        competition.columns = ['location', 'restaurant_count']
        top_locations = competition.head(top_n).copy()

        with st.spinner("Geocoding top locations..."):
            loc_tuple = tuple(top_locations['location'].tolist())
            top_locations['latitude'], top_locations['longitude'] = geocode_locations(loc_tuple)

        top_locations.dropna(subset=['latitude', 'longitude'], inplace=True)

        if top_locations.empty:
            st.error("Could not geocode any locations. Check network connection or input locations.")
            st.stop()

        demand_data = df.groupby('location').agg({'votes': 'sum', 'cleaned_rate': 'mean'}).reset_index()
        final_grid = pd.merge(top_locations, demand_data, on='location')

        final_grid['norm_demand'] = final_grid['votes'] / (final_grid['votes'].max() or 1)
        final_grid['norm_competition'] = final_grid['restaurant_count'] / (final_grid['restaurant_count'].max() or 1)
        final_grid['Opportunity_Score'] = final_grid['norm_demand'] - final_grid['norm_competition']
        final_grid = final_grid.sort_values(by='Opportunity_Score', ascending=False)

        available_spots = len(final_grid)
        actual_clusters = min(num_clusters, available_spots)

        if actual_clusters < num_clusters:
            st.warning(f"Adjusted clusters from {num_clusters} to {actual_clusters} due to limited valid location points.")

        top_spots = final_grid.head(actual_clusters)[['latitude', 'longitude']]

        if actual_clusters > 0:
            kmeans = KMeans(n_clusters=actual_clusters, random_state=42, n_init=10)
            truck_spots = kmeans.fit(top_spots).cluster_centers_
        else:
            truck_spots = np.empty((0, 2))

        map_center = [final_grid['latitude'].iloc[0], final_grid['longitude'].iloc[0]]
        m = folium.Map(location=map_center, zoom_start=12, tiles='CartoDB positron')

        heat_data = [[row['latitude'], row['longitude'], row['restaurant_count']] for _, row in final_grid.iterrows()]
        HeatMap(heat_data, radius=25, blur=15, min_opacity=0.3).add_to(m)

        for idx, spot in enumerate(truck_spots):
            folium.Marker(
                location=[spot[0], spot[1]],
                popup=f"OPTIMIZED SITE #{idx+1}",
                icon=folium.Icon(color='green', icon='cutlery')
            ).add_to(m)

        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("Interactive Map & Clusters")
            st_folium(m, width=700, height=500, returned_objects=[])

        with col2:
            st.subheader("Recommended Hub Coordinates")
            if len(truck_spots) > 0:
                for idx, spot in enumerate(truck_spots):
                    st.success(f"**Hub #{idx+1}:** `{spot[0]:.4f}° N, {spot[1]:.4f}° E`")
            else:
                st.info("No clusters generated.")

        st.subheader("Top Opportunity Hotspots")
        st.dataframe(
            final_grid[['location', 'restaurant_count', 'votes', 'Opportunity_Score']],
            use_container_width=True
        )

    except Exception as e:
        st.error(f"Error processing file: {e}")
else:
    st.warning("Please upload a CSV file in the sidebar to begin analysis.")