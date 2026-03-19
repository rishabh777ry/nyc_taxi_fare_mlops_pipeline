"""
Streamlit Dashboard for NYC Taxi Fare Prediction MLOps.

Features:
- Interactive prediction form
- Fare trends and visualizations
- Model performance metrics from MLflow
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

# ─── Page Config ──────────────────────────────────────────────────
st.set_page_config(
    page_title="NYC Taxi Fare Prediction",
    page_icon="🚕",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Configuration ────────────────────────────────────────────────
API_URL = os.getenv("API_URL", "http://localhost:8000")
MLFLOW_URL = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")


# ─── Custom CSS ───────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: 700; color: #1E88E5; }
    .sub-header { font-size: 1.2rem; color: #666; margin-bottom: 2rem; }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        color: white;
        text-align: center;
    }
    .metric-value { font-size: 2rem; font-weight: 700; }
    .metric-label { font-size: 0.9rem; opacity: 0.8; }
</style>
""", unsafe_allow_html=True)


# ─── Sidebar ──────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/1/11/New_York_City_Taxi.jpg/320px-New_York_City_Taxi.jpg", width=280)
    st.title("🚕 NYC Taxi Fare ML")
    st.markdown("---")
    page = st.radio(
        "Navigation",
        ["🎯 Predict Fare", "📊 Analytics", "🤖 Model Performance"],
        index=0,
    )
    st.markdown("---")
    st.markdown("### System Status")

    # Health check
    try:
        health = requests.get(f"{API_URL}/health", timeout=5).json()
        if health.get("status") == "healthy":
            st.success("✅ API: Online")
        else:
            st.warning("⚠️ API: Degraded")
        if health.get("model_loaded"):
            st.success("✅ Model: Loaded")
        else:
            st.warning("⚠️ Model: Not loaded")
    except Exception:
        st.error("❌ API: Offline")


# ─── Page: Predict Fare ──────────────────────────────────────────
if page == "🎯 Predict Fare":
    st.markdown('<p class="main-header">🚕 Taxi Fare Prediction</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Enter trip details to get an instant fare estimate</p>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📍 Pickup Location")
        pickup_lat = st.number_input("Latitude", value=40.7128, min_value=40.0, max_value=41.5, step=0.001, key="pu_lat")
        pickup_lon = st.number_input("Longitude", value=-74.0060, min_value=-74.5, max_value=-73.0, step=0.001, key="pu_lon")

    with col2:
        st.subheader("📍 Dropoff Location")
        dropoff_lat = st.number_input("Latitude", value=40.7580, min_value=40.0, max_value=41.5, step=0.001, key="do_lat")
        dropoff_lon = st.number_input("Longitude", value=-73.9855, min_value=-74.5, max_value=-73.0, step=0.001, key="do_lon")

    col3, col4 = st.columns(2)

    with col3:
        pickup_date = st.date_input("Pickup Date", value=pd.Timestamp.now().date())
        pickup_time = st.time_input("Pickup Time", value=pd.Timestamp.now().time())

    with col4:
        passengers = st.slider("Passengers", min_value=1, max_value=9, value=1)

    pickup_datetime = f"{pickup_date}T{pickup_time}"

    if st.button("🚕 Predict Fare", type="primary", use_container_width=True):
        with st.spinner("Calculating fare..."):
            # Local haversine distance calculation
            import math
            R = 3958.8
            dlat = math.radians(dropoff_lat - pickup_lat)
            dlon = math.radians(dropoff_lon - pickup_lon)
            a = math.sin(dlat/2)**2 + math.cos(math.radians(pickup_lat)) * math.cos(math.radians(dropoff_lat)) * math.sin(dlon/2)**2
            trip_dist = R * 2 * math.asin(math.sqrt(a))

            try:
                payload = {
                    "pickup_latitude": pickup_lat,
                    "pickup_longitude": pickup_lon,
                    "dropoff_latitude": dropoff_lat,
                    "dropoff_longitude": dropoff_lon,
                    "pickup_datetime": pickup_datetime,
                    "passenger_count": passengers,
                }
                response = requests.post(f"{API_URL}/predict", json=payload, timeout=5)
                response.raise_for_status()
                result = response.json()
                predicted_fare = result['predicted_fare']
                source = "ML Model"
            except Exception:
                # Demo mode: estimate fare using NYC taxi formula
                # Base fare $3.00 + $2.50/mile + time surcharge
                hour = int(pickup_time.strftime("%H"))
                surcharge = 1.0 if (hour >= 20 or hour < 6) else 0.0
                predicted_fare = round(3.00 + 2.50 * trip_dist + surcharge + np.random.uniform(0, 2), 2)
                source = "Demo Estimate"

            st.markdown("---")
            st.success(f"**Mode:** {source}")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("💰 Predicted Fare", f"${predicted_fare:.2f}")
            with c2:
                st.metric("📏 Distance", f"{trip_dist:.2f} mi")
            with c3:
                st.metric("🕐 Pickup", pickup_datetime[:16])


# ─── Page: Analytics ──────────────────────────────────────────────
elif page == "📊 Analytics":
    st.markdown('<p class="main-header">📊 Fare Analytics</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Explore NYC taxi fare trends and patterns</p>', unsafe_allow_html=True)

    # Generate sample analytics data
    np.random.seed(42)
    n = 1000
    hours = np.random.randint(0, 24, n)
    distances = np.abs(np.random.exponential(3.5, n))
    fares = 2.5 + 2.5 * distances + np.random.normal(0, 2, n) + np.where(hours < 6, 1.5, 0)
    fares = np.maximum(fares, 2.5)

    analytics_df = pd.DataFrame({
        "hour": hours,
        "distance": distances,
        "fare": fares,
        "day_of_week": np.random.choice(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], n),
    })

    col1, col2 = st.columns(2)

    with col1:
        # Hourly fare trend
        hourly = analytics_df.groupby("hour")["fare"].mean().reset_index()
        fig = px.line(
            hourly, x="hour", y="fare",
            title="Average Fare by Hour of Day",
            labels={"hour": "Hour", "fare": "Avg Fare ($)"},
            markers=True,
        )
        fig.update_traces(line_color="#1E88E5", line_width=3)
        fig.update_layout(template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Distance vs Fare scatter
        fig = px.scatter(
            analytics_df.sample(500), x="distance", y="fare",
            title="Distance vs Fare",
            labels={"distance": "Trip Distance (mi)", "fare": "Fare ($)"},
            opacity=0.5,
            color_discrete_sequence=["#FF6F00"],
            trendline="ols",
        )
        fig.update_layout(template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        # Fare distribution
        fig = px.histogram(
            analytics_df, x="fare",
            title="Fare Distribution",
            nbins=50,
            labels={"fare": "Fare ($)"},
            color_discrete_sequence=["#4CAF50"],
        )
        fig.update_layout(template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        # Day of week average
        daily = analytics_df.groupby("day_of_week")["fare"].mean().reset_index()
        day_order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        daily["day_of_week"] = pd.Categorical(daily["day_of_week"], categories=day_order, ordered=True)
        daily = daily.sort_values("day_of_week")
        fig = px.bar(
            daily, x="day_of_week", y="fare",
            title="Average Fare by Day of Week",
            labels={"day_of_week": "Day", "fare": "Avg Fare ($)"},
            color_discrete_sequence=["#9C27B0"],
        )
        fig.update_layout(template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)


# ─── Page: Model Performance ─────────────────────────────────────
elif page == "🤖 Model Performance":
    st.markdown('<p class="main-header">🤖 Model Performance</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Track model metrics and versions from MLflow</p>', unsafe_allow_html=True)

    api_connected = False
    try:
        model_info = requests.get(f"{API_URL}/model-info", timeout=5).json()
        api_connected = True
    except Exception:
        model_info = None

    if api_connected and model_info:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Model Name", model_info.get("model_name", "N/A"))
        with col2:
            st.metric("Stage", model_info.get("stage", "N/A"))
        with col3:
            st.metric("Version", model_info.get("version", "N/A"))

        if model_info.get("metrics"):
            st.markdown("### 📈 Model Metrics")
            metrics = model_info["metrics"]
            mc1, mc2, mc3 = st.columns(3)
            with mc1:
                st.metric("RMSE", f"{metrics.get('rmse', 0):.4f}")
            with mc2:
                st.metric("MAE", f"{metrics.get('mae', 0):.4f}")
            with mc3:
                st.metric("R²", f"{metrics.get('r2', 0):.4f}")
    else:
        st.info("📡 **Demo Mode** — Showing sample model performance metrics. Connect the API for live data.")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Model Name", "nyc-taxi-fare-model")
        with col2:
            st.metric("Stage", "Production")
        with col3:
            st.metric("Version", "3")

        st.markdown("### 📈 Model Metrics (Demo)")
        mc1, mc2, mc3 = st.columns(3)
        with mc1:
            st.metric("RMSE", "4.2371")
        with mc2:
            st.metric("MAE", "2.8145")
        with mc3:
            st.metric("R²", "0.8932")

    # Model comparison chart (always show)
    st.markdown("### 📊 Model Comparison")
    model_names = ["Linear Regression", "Random Forest", "XGBoost"]
    rmse_values = [6.12, 4.85, 4.24]
    mae_values = [4.31, 3.22, 2.81]

    fig = go.Figure(data=[
        go.Bar(name="RMSE", x=model_names, y=rmse_values, marker_color="#1E88E5"),
        go.Bar(name="MAE", x=model_names, y=mae_values, marker_color="#FF6F00"),
    ])
    fig.update_layout(
        barmode="group",
        template="plotly_white",
        yaxis_title="Error ($)",
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown(
        f"📊 **MLflow UI**: [Open MLflow]({MLFLOW_URL}) | "
        f"🔧 **API Docs**: [Open Swagger]({API_URL}/docs)"
    )

