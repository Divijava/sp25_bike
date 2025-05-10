import streamlit as st
import pandas as pd
import pytz
from datetime import datetime
from pathlib import Path
import sys
from streamlit_folium import st_folium
import folium
import plotly.express as px

# Setup imports
sys.path.append(str(Path(__file__).parent.parent))
from src.config import DATA_DIR
from src.inference import fetch_next_hour_predictions, load_batch_of_features_from_store
from src.plot_utils import plot_prediction


# App config
st.set_page_config(page_title="🚲 Citi Bike Forecast", layout="wide")

# Header
nyc_tz = pytz.timezone("America/New_York")
current_time = pd.Timestamp.now(tz="UTC").tz_convert(nyc_tz)
st.title("🚲 Citi Bike Demand Prediction")
st.markdown(f"**Prediction Time:** {current_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")

# Sidebar inputs
with st.sidebar:
    st.header("Prediction Controls")
    station_id = st.selectbox("Select Station ID:", options=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    progress_bar = st.progress(0)

# Load features and predictions
progress_bar.progress(0.2)
features = load_batch_of_features_from_store(current_time)
progress_bar.progress(0.5)
predictions = fetch_next_hour_predictions()
progress_bar.progress(0.8)

# Map
st.subheader("📍 Predicted Demand Map (NYC)")
map_df = predictions.copy()
map_df['Latitude'] = 40.75 + (map_df.index % 10) * 0.01  # Simulated lat/lon
map_df['Longitude'] = -73.99 + (map_df.index % 10) * 0.01

m = folium.Map(location=[40.75, -73.99], zoom_start=12)
for _, row in map_df.iterrows():
    folium.CircleMarker(
        location=[row['Latitude'], row['Longitude']],
        radius=row['predicted_demand'] / 10,
        popup=f"Zone {row['pickup_location_id']}<br>Predicted: {row['predicted_demand']:.1f}",
        color="blue",
        fill=True,
    ).add_to(m)

st_folium(m, width=800, height=600)
progress_bar.progress(1.0)

# Metrics
st.subheader("📊 Prediction Stats")
col1, col2, col3 = st.columns(3)
col1.metric("Max", f"{predictions['predicted_demand'].max():.0f}")
col2.metric("Min", f"{predictions['predicted_demand'].min():.0f}")
col3.metric("Avg", f"{predictions['predicted_demand'].mean():.0f}")

# Top 10 plot
st.subheader("📈 Top 10 Zones by Demand")
top10 = predictions.sort_values("predicted_demand", ascending=False).head(10)
st.dataframe(top10)

for location_id in top10["pickup_location_id"]:
    fig = plot_prediction(
        features=features[features["pickup_location_id"] == location_id],
        prediction=predictions[predictions["pickup_location_id"] == location_id]
    )
    st.plotly_chart(fig, use_container_width=True)
