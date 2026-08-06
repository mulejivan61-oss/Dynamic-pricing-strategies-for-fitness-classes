"""
Fitness Class Dynamic Pricing & Demand Forecasting Dashboard
Run: streamlit run app.py

Uses:
- best_rf_model.pkl  -> predicts "Number Booked" for a given class config
- arima_model.pkl    -> forecasts future total daily bookings
- rule-based function -> dynamic price suggestion (from notebook 06)
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="Fitness Dynamic Pricing", layout="wide")

# ------------------------------------------------------------------
# Load models (cached so they don't reload every interaction)
# ------------------------------------------------------------------
@st.cache_resource
def load_models():
    rf = joblib.load("best_rf_model.pkl")
    arima = joblib.load("arima_model.pkl")
    return rf, arima

rf_model, arima_model = load_models()

RF_COLUMNS = list(rf_model.feature_names_in_)

# Dropdown options extracted from the trained model's one-hot columns
TIMESLOTS = ["Afternoon", "Evening", "Morning"]          # Afternoon = baseline (dropped in training)
MONTHS = ["April", "June", "May"]                         # April = baseline
DAYS = ["Friday", "Monday", "Saturday", "Sunday", "Thursday", "Tuesday", "Wednesday"]  # Friday = baseline
SITES = ["BRP", "HXP", "NBL", "SBP", "TSC"]                # BRP = baseline
CLASSES = sorted(set(c.replace("ActivityDescription_", "")
                      for c in RF_COLUMNS if c.startswith("ActivityDescription_")))

# Premium classes (from notebook's rule-based pricing logic)
PREMIUM_CLASSES = [
    "Body Combat 11-12pm",
    "Pilates 9.30-10.30am",
    "Zumba 6.15-7.15pm",
    "Step 6-7pm",
    "Pilates 9.00-9.50am",
]

st.title("🏋️ Fitness Class Dynamic Pricing Dashboard")
st.caption("Predict bookings, forecast demand, and get a suggested dynamic price for a class.")

tab1, tab2, tab3 = st.tabs(["📊 Booking Predictor", "📈 Demand Forecast", "💰 Dynamic Price Calculator"])

# ====================================================================
# TAB 1: Booking Predictor (Random Forest)
# ====================================================================
with tab1:
    st.subheader("Predict Number of Bookings")
    col1, col2 = st.columns(2)

    with col1:
        price = st.number_input("Price (INR)", min_value=50, max_value=5000, value=499, step=10)
        max_bookees = st.number_input("Max Capacity (MaxBookees)", min_value=1, max_value=200, value=35)
        hour = st.slider("Hour of class (24h)", 0, 23, 9)
        timeslot = st.selectbox("Time Slot", TIMESLOTS)

    with col2:
        month = st.selectbox("Month", MONTHS)
        day = st.selectbox("Day of Week", DAYS)
        site = st.selectbox("Location (ActivitySiteID)", SITES)
        activity = st.selectbox("Class (ActivityDescription)", CLASSES)

    if st.button("Predict Bookings", type="primary"):
        # Build a single-row input matching training format
        row = pd.DataFrame([{
            "Price (INR)": price,
            "MaxBookees": max_bookees,
            "Hour": hour,
            "TimeSlot": timeslot,
            "Month": month,
            "Day": day,
            "ActivitySiteID": site,
            "ActivityDescription": activity,
        }])

        row_encoded = pd.get_dummies(
            row, columns=["TimeSlot", "Month", "Day", "ActivitySiteID", "ActivityDescription"]
        )
        # Align to the exact columns the model was trained on (missing -> 0)
        row_final = row_encoded.reindex(columns=RF_COLUMNS, fill_value=0)

        predicted_bookings = rf_model.predict(row_final)[0]
        predicted_bookings = max(0, min(predicted_bookings, max_bookees))
        occupancy = (predicted_bookings / max_bookees) * 100

        c1, c2, c3 = st.columns(3)
        c1.metric("Predicted Bookings", f"{predicted_bookings:.0f}")
        c2.metric("Predicted Occupancy", f"{occupancy:.1f}%")
        c3.metric("Predicted Revenue", f"₹{predicted_bookings * price:,.0f}")

# ====================================================================
# TAB 2: Demand Forecast (ARIMA)
# ====================================================================
with tab2:
    st.subheader("Forecast Future Total Daily Bookings")
    st.caption("Based on historical daily demand pattern (ARIMA model).")

    horizon = st.slider("Forecast horizon (days)", 5, 30, 10)

    if st.button("Run Forecast"):
        forecast = arima_model.forecast(steps=horizon)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(range(1, horizon + 1), forecast.values, marker="o", color="orange")
        ax.set_xlabel("Days Ahead")
        ax.set_ylabel("Predicted Total Bookings")
        ax.set_title(f"Demand Forecast — Next {horizon} Days")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

        st.dataframe(
            pd.DataFrame({"Day Ahead": range(1, horizon + 1), "Forecasted Bookings": forecast.values.round(1)}),
            hide_index=True,
        )

# ====================================================================
# TAB 3: Dynamic Price Calculator (rule-based, from notebook 06)
# ====================================================================
with tab3:
    st.subheader("Dynamic Price Suggestion")
    st.caption("Rule-based pricing adjustment using occupancy, time slot, day, class type, and location.")

    col1, col2 = st.columns(2)
    with col1:
        base_price = st.number_input("Base Price (INR)", min_value=50, max_value=5000, value=499, step=10, key="dp_price")
        occupancy_rate = st.slider("Current Occupancy Rate (%)", 0, 150, 80)
        dp_timeslot = st.selectbox("Time Slot", TIMESLOTS, key="dp_ts")

    with col2:
        dp_day = st.selectbox("Day of Week", DAYS, key="dp_day")
        dp_activity = st.selectbox("Class", CLASSES, key="dp_activity")
        dp_site = st.selectbox("Location", SITES, key="dp_site")

    def dynamic_pricing(price, occupancy, timeslot, day, activity, site):
        # Rule 1: Occupancy
        if occupancy >= 90:
            price *= 1.05
        elif occupancy < 40:
            price *= 0.95
        # Rule 2: Time Slot
        if timeslot == "Morning":
            price *= 1.03
        elif timeslot == "Evening":
            price *= 1.02
        elif timeslot == "Afternoon":
            price *= 0.97
        # Rule 3: Day
        if day == "Tuesday":
            price *= 1.03
        elif day == "Wednesday":
            price *= 1.02
        elif day == "Sunday":
            price *= 0.97
        # Rule 4: Premium class
        if activity in PREMIUM_CLASSES:
            price *= 1.05
        # Rule 5: Location
        if site == "HXP":
            price *= 1.03
        elif site == "BRP":
            price *= 1.02
        return round(price)

    if st.button("Calculate Dynamic Price", type="primary"):
        new_price = dynamic_pricing(base_price, occupancy_rate, dp_timeslot, dp_day, dp_activity, dp_site)
        change_pct = ((new_price - base_price) / base_price) * 100

        c1, c2, c3 = st.columns(3)
        c1.metric("Base Price", f"₹{base_price}")
        c2.metric("Suggested Price", f"₹{new_price}", f"{change_pct:+.1f}%")
        c3.metric("Occupancy Used", f"{occupancy_rate}%")

st.divider()
st.caption("⚠️ Predictions are estimates based on historical data (2018 fitness class bookings) — use as decision support, not absolute truth.")
