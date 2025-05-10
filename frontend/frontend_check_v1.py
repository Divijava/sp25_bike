import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import pytz
from datetime import datetime
from src.config import DATA_DIR
from src.inference import fetch_next_hour_predictions, load_batch_of_features_from_store
from src.plot_utils import plot_prediction

# Page setup
st.set_page_config(page_title="🚲 Citi Bike Demand Forecast", layout="wide")

# Time info
nyc_tz = pytz.timezone("America/New_York")
current_time = pd.Timestamp.now(tz="UTC").tz_convert(nyc_tz)
st.title("🚲 Citi Bike Demand Prediction")
st.markdown(f"**Prediction Time:** {current_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")

# Sidebar controls
with st.sidebar:
    st.header("Prediction Controls")
    top_n = st.selectbox("Select number of top locations by rides", options=list(range(1, 21)), index=2)
    progress = st.progress(0)

# Load features and predictions
progress.progress(0.3)
features = load_batch_of_features_from_store(current_time)
progress.progress(0.6)
predictions = fetch_next_hour_predictions()
progress.progress(1.0)

# Prediction Stats
st.subheader("📊 Prediction Stats")
col1, col2, col3 = st.columns(3)
col1.metric("Max Rides", f"{predictions['predicted_demand'].max():.0f}")
col2.metric("Min Rides", f"{predictions['predicted_demand'].min():.0f}")
col3.metric("Avg Rides", f"{predictions['predicted_demand'].mean():.0f}")

# Top N Predictions
st.subheader(f"📈 Top {top_n} Zones by Predicted Demand")
top_zones = predictions.sort_values("predicted_demand", ascending=False).head(top_n)
st.dataframe(top_zones)

# Plot predictions
for zone_id in top_zones["pickup_location_id"]:
    st.plotly_chart(
        plot_prediction(
            features=features[features["pickup_location_id"] == zone_id],
            prediction=predictions[predictions["pickup_location_id"] == zone_id]
        ),
        use_container_width=True
    )
