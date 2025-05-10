import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
from src.inference import fetch_predictions, fetch_hourly_rides

# Page Setup
st.set_page_config(page_title="🚲 Citi Bike Monitoring", layout="wide")
st.markdown("""
    <style>
    .big-metric { font-size: 22px; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center;'>📉 Citi Bike Demand Monitoring</h1>", unsafe_allow_html=True)
st.caption("Last refresh: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

# Sidebar Controls
with st.sidebar:
    st.header("⚙️ Settings")
    past_hours = st.slider("⏱ Lookback window (hours):", 6, 72, 24)
    st.success("Data loaded from last {} hours".format(past_hours))
    st.button("🔁 Refresh Dashboard", help="Click to rerun app manually (Streamlit auto-refreshes on change)")

# Load data
actual_df = fetch_hourly_rides(past_hours)
pred_df = fetch_predictions(past_hours)
merged = actual_df.merge(pred_df, on=["pickup_hour", "pickup_location_id"])
merged["abs_error"] = abs(merged["rides"] - merged["predicted_demand"])
merged["squared_error"] = (merged["rides"] - merged["predicted_demand"]) ** 2

# Filter for valid MAPE calc
non_zero_actuals = merged[merged["rides"] != 0].copy()
non_zero_actuals["ape"] = abs((non_zero_actuals["rides"] - non_zero_actuals["predicted_demand"]) / non_zero_actuals["rides"]) * 100

# Metrics
mae = merged["abs_error"].mean()
rmse = (merged["squared_error"].mean()) ** 0.5
mape = non_zero_actuals["ape"].mean() if not non_zero_actuals.empty else None

# Metric Display
st.markdown("### 📊 Current Model Stats")
col1, col2, col3 = st.columns(3)
col1.metric("📈 MAE", f"{mae:.2f}")
col2.metric("📉 MAPE", f"{mape:.2f}%" if mape else "N/A")
col3.metric("📏 RMSE", f"{rmse:.2f}")

st.markdown("---")

# Line Chart
st.markdown("### 📈 MAE Trend Over Time")
hourly_mae = merged.groupby("pickup_hour")["abs_error"].mean().reset_index()
fig = px.line(
    hourly_mae,
    x="pickup_hour",
    y="abs_error",
    title="Hourly Mean Absolute Error",
    markers=True
)
fig.update_traces(line=dict(color="royalblue", width=3), marker=dict(size=7))
fig.update_layout(title_font_size=18, xaxis_title="Time", yaxis_title="MAE")
st.plotly_chart(fig, use_container_width=True)

# Raw Table View
st.markdown("### 🔍 Highest Prediction Errors")
top_errors = merged.sort_values("abs_error", ascending=False).head(15)[
    ["pickup_hour", "pickup_location_id", "rides", "predicted_demand", "abs_error", "squared_error"]
]

with st.expander("📋 View Top 15 Errors", expanded=True):
    st.dataframe(top_errors.style.background_gradient(cmap="Reds", subset=["abs_error"]))

# CSV download
csv = top_errors.to_csv(index=False).encode("utf-8")
st.download_button(
    label="💾 Download Top Errors as CSV",
    data=csv,
    file_name="top_prediction_errors.csv",
    mime="text/csv",
)
