
"""
Fitness Class Dynamic Pricing & Demand Intelligence Dashboard
Run: streamlit run app.py
 
Ties together the three core components built during modeling:
  1. Price Elasticity        (notebook 03) -> price-vs-demand curve, elasticity coefficient
  2. Demand Forecasting       (notebook 04) -> ARIMA total daily bookings forecast
  3. Dynamic Pricing Algorithm(notebook 06) -> ML-driven revenue-maximizing price
     (replaces arbitrary rule multipliers with an actual optimization over the
      trained RF model's demand response)
 
Models used:
- best_rf_model.pkl -> predicts "Number Booked" for a given class configuration
- arima_model.pkl   -> forecasts future total daily bookings
 
Optional data file (used only for the Model Insights validation tab):
- fitness_classes_EDA.csv
"""
 
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
 
st.set_page_config(page_title="Fitness Dynamic Pricing", layout="wide", page_icon="🏋️")
 
# ------------------------------------------------------------------
# Load models
# ------------------------------------------------------------------
@st.cache_resource
def load_models():
    rf = joblib.load("best_rf_model.pkl")
    try:
        arima = joblib.load("arima_model.pkl")
    except Exception:
        arima = None
    return rf, arima
 
 
try:
    rf_model, arima_model = load_models()
except FileNotFoundError as e:
    st.error(f"Model file not found: {e}. Keep best_rf_model.pkl / arima_model.pkl in the app folder.")
    st.stop()
 
RF_COLUMNS = list(rf_model.feature_names_in_)
 
# Dropdown options (baseline categories are dropped in one-hot training, kept here so user can still pick them)
TIMESLOTS = ["Afternoon", "Evening", "Morning"]
MONTHS = ["April", "June", "May"]
DAYS = ["Friday", "Monday", "Saturday", "Sunday", "Thursday", "Tuesday", "Wednesday"]
SITES = ["BRP", "HXP", "NBL", "SBP", "TSC"]
CLASSES = sorted(set(c.replace("ActivityDescription_", "")
                      for c in RF_COLUMNS if c.startswith("ActivityDescription_")))
 
PREMIUM_HINT_CLASSES = [
    "Body Combat 11-12pm", "Pilates 9.30-10.30am", "Zumba 6.15-7.15pm",
    "Step 6-7pm", "Pilates 9.00-9.50am",
]
 
 
# ------------------------------------------------------------------
# Core helpers — shared by every tab so predictor / forecaster / optimizer
# all speak to the SAME model instead of drifting apart
# ------------------------------------------------------------------
def build_row(price, max_bookees, hour, timeslot, month, day, site, activity):
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
    row_encoded = pd.get_dummies(row, columns=["TimeSlot", "Month", "Day", "ActivitySiteID", "ActivityDescription"])
    return row_encoded.reindex(columns=RF_COLUMNS, fill_value=0)
 
 
def predict_bookings(price, max_bookees, hour, timeslot, month, day, site, activity):
    row_final = build_row(price, max_bookees, hour, timeslot, month, day, site, activity)
    pred = rf_model.predict(row_final)[0]
    return float(np.clip(pred, 0, max_bookees))
 
 
def find_optimal_price(max_bookees, hour, timeslot, month, day, site, activity,
                        price_min=100, price_max=2000, step=10):
    prices = np.arange(price_min, price_max + step, step)
    bookings = [predict_bookings(p, max_bookees, hour, timeslot, month, day, site, activity) for p in prices]
    revenue = prices * np.array(bookings)
    curve = pd.DataFrame({"Price": prices, "PredictedBookings": bookings, "Revenue": revenue})
    curve["Occupancy%"] = (curve["PredictedBookings"] / max_bookees) * 100
    best_row = curve.loc[curve["Revenue"].idxmax()]
    return curve, best_row
 
 
def price_elasticity(curve, base_price):
    """Point elasticity of demand estimated straight from the RF model's own price sweep."""
    nearest = curve.iloc[(curve["Price"] - base_price).abs().argsort()[:2]].sort_values("Price")
    if len(nearest) < 2:
        return None
    p1, p2 = nearest["Price"].values
    b1, b2 = nearest["PredictedBookings"].values
    if p1 == 0 or b1 == 0 or p2 == p1:
        return None
    pct_change_q = (b2 - b1) / b1
    pct_change_p = (p2 - p1) / p1
    if pct_change_p == 0:
        return None
    return pct_change_q / pct_change_p
 
 
st.title("🏋️ Fitness Class Dynamic Pricing & Demand Intelligence")
st.caption("ML-driven booking prediction, demand forecasting, and revenue-maximizing price optimization.")
 
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Booking Predictor",
    "📈 Demand Forecast",
    "🎯 Optimal Price Finder",
    "📦 Batch Optimizer",
    "🔍 Model Insights",
])
 
# ====================================================================
# TAB 1: Booking Predictor (Random Forest)
# ====================================================================
with tab1:
    st.subheader("Predict Bookings for a Given Price")
    col1, col2 = st.columns(2)
    with col1:
        price = st.number_input("Price (INR)", 50, 5000, 499, 10, key="t1_price")
        max_bookees = st.number_input("Max Capacity (MaxBookees)", 1, 200, 35, key="t1_cap")
        hour = st.slider("Hour of class (24h)", 0, 23, 9, key="t1_hour")
        timeslot = st.selectbox("Time Slot", TIMESLOTS, key="t1_ts")
    with col2:
        month = st.selectbox("Month", MONTHS, key="t1_month")
        day = st.selectbox("Day of Week", DAYS, key="t1_day")
        site = st.selectbox("Location (ActivitySiteID)", SITES, key="t1_site")
        activity = st.selectbox("Class (ActivityDescription)", CLASSES, key="t1_activity")
 
    if st.button("Predict Bookings", type="primary", key="t1_btn"):
        pred = predict_bookings(price, max_bookees, hour, timeslot, month, day, site, activity)
        occupancy = (pred / max_bookees) * 100
        c1, c2, c3 = st.columns(3)
        c1.metric("Predicted Bookings", f"{pred:.0f}")
        c2.metric("Predicted Occupancy", f"{occupancy:.1f}%")
        c3.metric("Predicted Revenue", f"₹{pred * price:,.0f}")
 
# ====================================================================
# TAB 2: Demand Forecast (ARIMA) + Revenue Projection
# ====================================================================
with tab2:
    st.subheader("Forecast Future Demand & Revenue")
    if arima_model is None:
        st.warning("ARIMA model not loaded — check that arima_model.pkl is in the app folder.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            horizon = st.slider("Forecast horizon (days)", 5, 30, 10)
        with col2:
            avg_price = st.number_input("Assumed avg. price per booking (INR)", 50, 5000, 499, 10)
 
        if st.button("Run Forecast", type="primary"):
            forecast = arima_model.forecast(steps=horizon)
            fc_values = np.maximum(np.asarray(forecast.values, dtype=float), 0)
            revenue_proj = fc_values * avg_price
 
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=list(range(1, horizon + 1)), y=fc_values,
                                      mode="lines+markers", name="Predicted Bookings",
                                      line=dict(color="orange")))
            fig.update_layout(title=f"Booking Demand Forecast — Next {horizon} Days",
                               xaxis_title="Days Ahead", yaxis_title="Predicted Total Bookings")
            st.plotly_chart(fig, use_container_width=True)
 
            c1, c2 = st.columns(2)
            c1.metric("Total Forecasted Bookings", f"{fc_values.sum():.0f}")
            c2.metric("Projected Revenue (at assumed price)", f"₹{revenue_proj.sum():,.0f}")
 
            st.dataframe(pd.DataFrame({
                "Day Ahead": range(1, horizon + 1),
                "Forecasted Bookings": fc_values.round(1),
                "Projected Revenue (₹)": revenue_proj.round(0),
            }), hide_index=True, use_container_width=True)
 
# ====================================================================
# TAB 3: Optimal Price Finder — ML-driven (replaces old hardcoded rules)
# ====================================================================
with tab3:
    st.subheader("Find the Revenue-Maximizing Price")
    st.caption("Sweeps a price range through the trained RF model and finds where "
               "Price × Predicted Bookings (revenue) peaks — the model IS the pricing engine, "
               "not a fixed rule table.")
 
    col1, col2 = st.columns(2)
    with col1:
        opt_cap = st.number_input("Max Capacity", 1, 200, 35, key="t3_cap")
        opt_hour = st.slider("Hour of class (24h)", 0, 23, 9, key="t3_hour")
        opt_ts = st.selectbox("Time Slot", TIMESLOTS, key="t3_ts")
        opt_month = st.selectbox("Month", MONTHS, key="t3_month")
    with col2:
        opt_day = st.selectbox("Day of Week", DAYS, key="t3_day")
        opt_site = st.selectbox("Location", SITES, key="t3_site")
        opt_activity = st.selectbox("Class", CLASSES, key="t3_activity")
        base_price = st.number_input("Current / base price (INR)", 50, 5000, 499, 10, key="t3_base")
 
    price_range = st.slider("Price range to search (INR)", 50, 3000, (100, 1200), step=10)
 
    if opt_activity in PREMIUM_HINT_CLASSES:
        st.caption("ℹ️ This class was historically identified as high-demand/premium in the EDA — "
                   "expect the optimizer to push price upward if demand holds.")
 
    if st.button("Find Optimal Price", type="primary", key="t3_btn"):
        curve, best = find_optimal_price(opt_cap, opt_hour, opt_ts, opt_month, opt_day,
                                          opt_site, opt_activity, price_range[0], price_range[1])
        base_pred = predict_bookings(base_price, opt_cap, opt_hour, opt_ts, opt_month,
                                      opt_day, opt_site, opt_activity)
        base_revenue = base_price * base_pred
        uplift = ((best["Revenue"] - base_revenue) / base_revenue * 100) if base_revenue > 0 else 0
 
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Optimal Price", f"₹{best['Price']:.0f}")
        c2.metric("Expected Bookings", f"{best['PredictedBookings']:.0f}")
        c3.metric("Expected Revenue", f"₹{best['Revenue']:,.0f}")
        c4.metric("Revenue vs Current Price", f"{uplift:+.1f}%")
 
        elasticity = price_elasticity(curve, base_price)
        if elasticity is not None:
            label = "Elastic — demand is price-sensitive" if abs(elasticity) > 1 else "Inelastic — demand is fairly price-stable"
            st.info(f"📐 Estimated price elasticity of demand near ₹{base_price}: **{elasticity:.2f}** — {label}")
 
        fig1 = px.line(curve, x="Price", y="PredictedBookings", title="Price vs Predicted Bookings (Demand Curve)")
        fig1.add_vline(x=best["Price"], line_dash="dash", line_color="green", annotation_text="Optimal")
        fig1.add_vline(x=base_price, line_dash="dot", line_color="red", annotation_text="Current")
        st.plotly_chart(fig1, use_container_width=True)
 
        fig2 = px.line(curve, x="Price", y="Revenue", title="Price vs Expected Revenue")
        fig2.add_vline(x=best["Price"], line_dash="dash", line_color="green", annotation_text="Optimal")
        fig2.add_vline(x=base_price, line_dash="dot", line_color="red", annotation_text="Current")
        st.plotly_chart(fig2, use_container_width=True)
 
# ====================================================================
# TAB 4: Batch Optimizer — price multiple classes in one shot
# ====================================================================
with tab4:
    st.subheader("Optimize Prices for Multiple Classes at Once")
    st.caption("Upload a CSV with columns: MaxBookees, Hour, TimeSlot, Month, Day, ActivitySiteID, "
               "ActivityDescription, CurrentPrice — get the recommended optimal price per row.")
 
    template = pd.DataFrame([{
        "MaxBookees": 35, "Hour": 9, "TimeSlot": "Morning", "Month": "June",
        "Day": "Monday", "ActivitySiteID": "BRP", "ActivityDescription": CLASSES[0],
        "CurrentPrice": 499,
    }])
    st.download_button("Download CSV template", template.to_csv(index=False),
                        "price_optimizer_template.csv", "text/csv")
 
    uploaded = st.file_uploader("Upload classes CSV", type="csv")
    if uploaded is not None:
        batch = pd.read_csv(uploaded)
        results = []
        with st.spinner("Optimizing prices for every row..."):
            for _, r in batch.iterrows():
                curve, best = find_optimal_price(
                    r["MaxBookees"], r["Hour"], r["TimeSlot"], r["Month"], r["Day"],
                    r["ActivitySiteID"], r["ActivityDescription"], 50, 2000, 20,
                )
                base_pred = predict_bookings(r["CurrentPrice"], r["MaxBookees"], r["Hour"],
                                              r["TimeSlot"], r["Month"], r["Day"],
                                              r["ActivitySiteID"], r["ActivityDescription"])
                base_rev = r["CurrentPrice"] * base_pred
                uplift = ((best["Revenue"] - base_rev) / base_rev * 100) if base_rev > 0 else 0
                results.append({
                    **r.to_dict(),
                    "RecommendedPrice": round(best["Price"]),
                    "ExpectedBookings": round(best["PredictedBookings"]),
                    "ExpectedRevenue": round(best["Revenue"]),
                    "RevenueUplift%": round(uplift, 1),
                })
        result_df = pd.DataFrame(results)
        st.dataframe(result_df, use_container_width=True, hide_index=True)
        st.download_button("Download results CSV", result_df.to_csv(index=False),
                            "optimized_prices.csv", "text/csv")
        st.metric("Average Revenue Uplift Across Uploaded Classes", f"{result_df['RevenueUplift%'].mean():+.1f}%")
 
# ====================================================================
# TAB 5: Model Insights — explainability + validation
# ====================================================================
with tab5:
    st.subheader("Model Explainability & Validation")
 
    st.markdown("**Feature Importance (Random Forest)**")
    importances = pd.DataFrame({
        "Feature": RF_COLUMNS,
        "Importance": rf_model.feature_importances_,
    }).sort_values("Importance", ascending=False).head(15)
    fig_imp = px.bar(importances, x="Importance", y="Feature", orientation="h",
                      title="Top 15 Features Driving Booking Predictions")
    fig_imp.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig_imp, use_container_width=True)
 
    st.markdown("**Model Accuracy — Actual vs Predicted**")
    try:
        hist = pd.read_csv("fitness_classes_EDA.csv")
        required = {"Price (INR)", "MaxBookees", "Hour", "TimeSlot", "Month", "Day",
                     "ActivitySiteID", "ActivityDescription", "Number Booked"}
        if required.issubset(hist.columns):
            hist_sample = hist.sample(min(300, len(hist)), random_state=42)
            preds = [predict_bookings(row["Price (INR)"], row["MaxBookees"], row["Hour"],
                                       row["TimeSlot"], row["Month"], row["Day"],
                                       row["ActivitySiteID"], row["ActivityDescription"])
                     for _, row in hist_sample.iterrows()]
            actual = hist_sample["Number Booked"].values.astype(float)
            preds = np.array(preds)
            mae = np.mean(np.abs(preds - actual))
            rmse = np.sqrt(np.mean((preds - actual) ** 2))
            ss_res = np.sum((actual - preds) ** 2)
            ss_tot = np.sum((actual - actual.mean()) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
 
            c1, c2, c3 = st.columns(3)
            c1.metric("R²", f"{r2:.3f}")
            c2.metric("MAE", f"{mae:.2f}")
            c3.metric("RMSE", f"{rmse:.2f}")
 
            fig_val = px.scatter(x=actual, y=preds,
                                  labels={"x": "Actual Bookings", "y": "Predicted Bookings"},
                                  title="Actual vs Predicted Bookings (sample of 300)")
            fig_val.add_shape(type="line", x0=actual.min(), y0=actual.min(),
                               x1=actual.max(), y1=actual.max(), line=dict(dash="dash", color="red"))
            st.plotly_chart(fig_val, use_container_width=True)
        else:
            st.info("fitness_classes_EDA.csv is missing some required columns — validation skipped.")
    except FileNotFoundError:
        st.info("fitness_classes_EDA.csv not found in the app folder — validation section skipped.")
 
st.divider()
with st.expander("ℹ️ Methodology"):
    st.markdown("""
    - **Booking Predictor**: Random Forest regression trained on historical class booking data.
    - **Demand Forecast**: ARIMA time-series model on total daily bookings.
    - **Optimal Price Finder**: sweeps price through the RF model to find where
      `Price × Predicted Bookings` (revenue) is maximized, subject to capacity —
      this *is* the dynamic pricing algorithm, driven by the model instead of fixed multipliers.
    - **Elasticity**: computed directly from the RF model's own demand response around the
      current price, rather than assumed from a rule table.
    - **Batch Optimizer**: applies the same optimization to many classes at once for
      operational, portfolio-level pricing decisions.
    """)
st.caption("⚠️ Predictions are estimates based on historical data (2018 fitness class bookings) — "
           "use as decision support, not absolute truth.")
