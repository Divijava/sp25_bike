import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import pytz
from datetime import datetime
from streamlit_folium import st_folium
import folium
import plotly.express as px

from src.config import DATA_DIR
from src.inference import fetch_next_hour_predictions, load_batch_of_features_from_store
from src.plot_utils import plot_prediction

# Set up page
st.set_page_config(page_title="🚲 Citi Bike Demand Forecast", layout="wide")

# Time info
nyc_tz = pytz.timezone("America/New_York")
current_time = pd.Timestamp.now(tz="UTC").tz_convert(nyc_tz)
st.title("🚲 Citi Bike Demand Prediction")
st.markdown(f"**Prediction Time:** {current_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")

# Sidebar options
with st.sidebar:
    st.header("Prediction Controls")
    station_id = st.selectbox("Select Station ID:", options=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    progress = st.progress(0)

# Load data
progress.progress(0.2)
features = load_batch_of_features_from_store(current_time)
progress.progress(0.6)
predictions = fetch_next_hour_predictions()
progress.progress(1.0)

# Map Display
st.subheader("📍 Predicted Demand Map (NYC Zones)")
pred_map = predictions.copy()
pred_map['Latitude'] = 40.75 + (pred_map.index % 10) * 0.01
pred_map['Longitude'] = -73.99 + (pred_map.index % 10) * 0.01

m = folium.Map(location=[40.75, -73.99], zoom_start=12)
for _, row in pred_map.iterrows():
    folium.CircleMarker(
        location=[row["Latitude"], row["Longitude"]],
        radius=row["predicted_demand"] / 10,
        popup=f"Zone {row['pickup_location_id']}<br>{row['predicted_demand']:.1f} trips",
        color="blue",
        fill=True,
    ).add_to(m)
st_folium(m, width=800, height=500)

# Metrics
st.subheader("📊 Prediction Stats")
col1, col2, col3 = st.columns(3)
col1.metric("Max Rides", f"{predictions['predicted_demand'].max():.0f}")
col2.metric("Min Rides", f"{predictions['predicted_demand'].min():.0f}")
col3.metric("Avg Rides", f"{predictions['predicted_demand'].mean():.0f}")

# Top 10 predictions
st.subheader("📈 Top 10 Zones by Predicted Demand")
top10 = predictions.sort_values("predicted_demand", ascending=False).head(10)
st.dataframe(top10)

# Plots per zone
for zone_id in top10["pickup_location_id"]:
    st.plotly_chart(
        plot_prediction(
            features=features[features["pickup_location_id"] == zone_id],
            prediction=predictions[predictions["pickup_location_id"] == zone_id]
        ),
        use_container_width=True
    )
