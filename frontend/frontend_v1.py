import streamlit as st
import pandas as pd
import pytz
from datetime import datetime, timedelta
from streamlit_folium import st_folium

from src.config import DATA_DIR
from src.inference import (
    get_model_predictions,
    load_batch_of_features_from_store,
    load_model_from_registry,
    load_metrics_from_registry,
)
from src.plot_utils import plot_aggregated_time_series
from src.data_utils import load_shape_data_file, create_taxi_map

# NYC timezone
nyc_tz = pytz.timezone("America/New_York")
current_time_utc = datetime.utcnow().replace(tzinfo=pytz.utc)
current_time_nyc = current_time_utc.astimezone(nyc_tz)

# UI
st.set_page_config(layout="wide")
st.title("Citi Bike Demand Prediction (Next Hour)")
st.subheader(f"Current NYC Time: {current_time_nyc.strftime('%Y-%m-%d %H:%M:%S %Z')}")

# Progress bar
progress_bar = st.sidebar.progress(0)
N_STEPS = 5

# Step 1: Load NYC zone shapefile
with st.spinner("Loading NYC zone shapefile..."):
    shapefile = load_shape_data_file(DATA_DIR)
    st.sidebar.success("Shapefile loaded")
    progress_bar.progress(1 / N_STEPS)

# Step 2: Load features
with st.spinner("Loading features from Hopsworks..."):
    features = load_batch_of_features_from_store(current_time_utc)
    st.sidebar.success("Features loaded")
    progress_bar.progress(2 / N_STEPS)

# Step 3: Load model
with st.spinner("Loading model from registry..."):
    model = load_model_from_registry()
    st.sidebar.success("Model loaded")
    progress_bar.progress(3 / N_STEPS)

# Step 4: Predict
with st.spinner("Making predictions..."):
    predictions = get_model_predictions(model, features)
    predictions["pickup_hour"] = current_time_nyc.replace(minute=0, second=0, microsecond=0)
    st.sidebar.success("Predictions ready")
    progress_bar.progress(4 / N_STEPS)

# Step 5: Map + Plot
with st.spinner("Generating map and plots..."):
    st.subheader("📍 Predicted Demand Map")
    map_ = create_taxi_map(shapefile, predictions)
    st_folium(map_, width=800, height=600)

    st.subheader("📊 Prediction Stats")
    col1, col2, col3 = st.columns(3)
    col1.metric("Average Rides", f"{predictions['predicted_demand'].mean():.2f}")
    col2.metric("Max Rides", f"{predictions['predicted_demand'].max():.0f}")
    col3.metric("Min Rides", f"{predictions['predicted_demand'].min():.0f}")

    st.dataframe(predictions.sort_values("predicted_demand", ascending=False).head(10))
    top10 = (
        predictions.sort_values("predicted_demand", ascending=False).head(10)[
            "pickup_location_id"
        ].tolist()
    )
    for location_id in top10:
        fig = plot_aggregated_time_series(
            features=features,
            targets=predictions["predicted_demand"],
            row_id=location_id,
            predictions=predictions["predicted_demand"],
        )
        st.plotly_chart(fig, theme="streamlit", use_container_width=True)
    progress_bar.progress(1.0)
    st.success("App finished!")
