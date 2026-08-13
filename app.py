
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
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
 
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
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
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
 
 
def predict_bookings_batch(rows):
    """Predict many rows in ONE RF call instead of one call per row."""
    if not isinstance(rows, pd.DataFrame):
        rows = pd.DataFrame(rows)
    encoded = pd.get_dummies(
        rows,
        columns=["TimeSlot", "Month", "Day", "ActivitySiteID", "ActivityDescription"]
    ).reindex(columns=RF_COLUMNS, fill_value=0)
    preds = np.asarray(rf_model.predict(encoded), dtype=float)
    caps = rows["MaxBookees"].to_numpy(dtype=float)
    return np.clip(preds, 0, caps)


def show_dataset_health(df):
    """Show lightweight data-quality diagnostics for the currently uploaded CSV."""
    rows, cols = df.shape
    missing = int(df.isna().sum().sum())
    duplicates = int(df.duplicated().sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{rows:,}")
    c2.metric("Columns", f"{cols:,}")
    c3.metric("Missing Cells", f"{missing:,}")
    c4.metric("Duplicate Rows", f"{duplicates:,}")

    if missing == 0 and duplicates == 0:
        st.success("Dataset Health: Excellent — no missing cells or duplicate rows detected.")
    elif missing == 0:
        st.info(f"Dataset Health: Good — no missing cells; {duplicates:,} duplicate rows detected.")
    else:
        st.warning(f"Dataset Health: Review required — {missing:,} missing cells detected.")

    # Keep this visual compact so a 1000+ row upload does not make the app heavy.
    quality = pd.DataFrame({
        "Column": df.columns,
        "Missing": df.isna().sum().values,
        "Unique": [df[c].nunique(dropna=True) for c in df.columns]
    })
    quality = quality[quality["Missing"] > 0].sort_values("Missing", ascending=False)

    if not quality.empty:
        st.dataframe(quality, use_container_width=True, hide_index=True)


def evaluate_uploaded_data(df):
    """Calculate live metrics only when the uploaded file has a real target."""
    required = {
        "Number Booked", "Price (INR)", "MaxBookees", "Hour", "TimeSlot",
        "Month", "Day", "ActivitySiteID", "ActivityDescription"
    }
    if not required.issubset(df.columns):
        return None

    rows = df[[
        "Price (INR)", "MaxBookees", "Hour", "TimeSlot", "Month", "Day",
        "ActivitySiteID", "ActivityDescription"
    ]].copy()
    actual_series = pd.to_numeric(df["Number Booked"], errors="coerce")
    valid = actual_series.notna()
    rows = rows.loc[valid].reset_index(drop=True)
    actual = actual_series.loc[valid].to_numpy(dtype=float)

    if len(actual) < 2:
        return None

    predicted = predict_bookings_batch(rows)
    return {
        "r2": float(r2_score(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "mae": float(mean_absolute_error(actual, predicted)),
        "actual": actual,
        "predicted": predicted,
    }


def predict_bookings(price, max_bookees, hour, timeslot, month, day, site, activity):
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
    return float(predict_bookings_batch(row)[0])
 
 
def find_optimal_price(max_bookees, hour, timeslot, month, day, site, activity,
                        base_price=499, adjustment_pct=0.05, step=5):
    """RF demand/revenue optimization with a practical local price guardrail."""
    base_price = float(base_price)
    low = max(50, base_price * (1 - adjustment_pct))
    high = base_price * (1 + adjustment_pct)
    prices = np.arange(np.floor(low / step) * step, np.ceil(high / step) * step + step, step)
    prices = np.unique(np.append(prices, base_price))
    candidates = pd.DataFrame({
        "Price (INR)": prices, "MaxBookees": max_bookees, "Hour": hour,
        "TimeSlot": timeslot, "Month": month, "Day": day,
        "ActivitySiteID": site, "ActivityDescription": activity,
    })
    bookings = predict_bookings_batch(candidates)
    revenue = prices * bookings
    curve = pd.DataFrame({"Price": prices, "PredictedBookings": bookings, "Revenue": revenue})
    curve["Occupancy%"] = (curve["PredictedBookings"] / max_bookees) * 100
    return curve, curve.loc[curve["Revenue"].idxmax()]


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
 
 

def recommendation_reason(base_price, best_price, base_pred, max_bookees, timeslot, day, site, activity, elasticity):
    change = ((best_price - base_price) / base_price) * 100 if base_price else 0
    occupancy = (base_pred / max_bookees) * 100 if max_bookees else 0
    reasons = []
    if occupancy >= 90:
        reasons.append("high current occupancy")
    elif occupancy < 40:
        reasons.append("low current occupancy")
    if elasticity is not None:
        if elasticity < -1:
            reasons.append("elastic demand")
        elif elasticity > -0.2:
            reasons.append("relatively inelastic demand")
    if timeslot == "Morning": reasons.append("morning demand signal")
    elif timeslot == "Evening": reasons.append("evening demand signal")
    if day == "Tuesday": reasons.append("Tuesday demand signal")
    if site in {"HXP", "BRP"}: reasons.append(f"strong site signal ({site})")
    if activity in PREMIUM_HINT_CLASSES: reasons.append("historically high-demand class")
    reasons_text = ", ".join(reasons[:3]) if reasons else "RF-predicted revenue response"
    direction = "increase" if change > 0.05 else ("decrease" if change < -0.05 else "keep near current")
    return f"Recommended to {direction} the price by {abs(change):.1f}% based on {reasons_text}."


def recommendation_reason(base_price, best_price, base_pred, max_bookees, timeslot, day, site, activity, elasticity):
    change = ((best_price - base_price) / base_price) * 100 if base_price else 0
    occupancy = (base_pred / max_bookees) * 100 if max_bookees else 0
    reasons = []
    if occupancy >= 90: reasons.append("high current occupancy")
    elif occupancy < 40: reasons.append("low current occupancy")
    if elasticity is not None:
        if elasticity < -1: reasons.append("elastic demand")
        elif elasticity > -0.2: reasons.append("relatively inelastic demand")
    if timeslot == "Morning": reasons.append("morning demand signal")
    elif timeslot == "Evening": reasons.append("evening demand signal")
    if day == "Tuesday": reasons.append("Tuesday demand signal")
    if site in {"HXP", "BRP"}: reasons.append(f"strong site signal ({site})")
    if activity in PREMIUM_HINT_CLASSES: reasons.append("historically high-demand class")
    reasons_text = ", ".join(reasons[:3]) if reasons else "RF-predicted revenue response"
    direction = "increase" if change > 0.05 else ("decrease" if change < -0.05 else "keep near current")
    return f"Recommended to {direction} the price by {abs(change):.1f}% based on {reasons_text}."

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
 
    adjustment_pct = st.slider("Maximum price adjustment", 1, 10, 5, step=1, format="%d%%") / 100
 
    if opt_activity in PREMIUM_HINT_CLASSES:
        st.caption("ℹ️ This class was historically identified as high-demand/premium in the EDA — "
                   "expect the optimizer to push price upward if demand holds.")
 
    if st.button("Find Optimal Price", type="primary", key="t3_btn"):
        curve, best = find_optimal_price(opt_cap, opt_hour, opt_ts, opt_month, opt_day,
                                          opt_site, opt_activity, base_price, adjustment_pct, 5)
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

        st.success("💡 " + recommendation_reason(
            base_price, best["Price"], base_pred, opt_cap, opt_ts, opt_day,
            opt_site, opt_activity, elasticity
        ))
        st.caption(f"Guardrail: recommendation is limited to ±{adjustment_pct*100:.0f}% of the current price to avoid unrealistic extrapolation.")
 
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
               "ActivityDescription, CurrentPrice. Add Number Booked to enable live model evaluation.")

    template = pd.DataFrame([{
        "MaxBookees": 35, "Hour": 9, "TimeSlot": "Morning", "Month": "June",
        "Day": "Monday", "ActivitySiteID": "BRP", "ActivityDescription": CLASSES[0],
        "CurrentPrice": 499, "Number Booked": 22,
    }])
    st.download_button("Download CSV template", template.to_csv(index=False),
                       "price_optimizer_template.csv", "text/csv")

    uploaded = st.file_uploader(
        "Upload classes CSV",
        type="csv",
        help="Use the provided template. Batch optimization is vectorized for large uploads."
    )

    if uploaded is not None:
        if uploaded.size > 10 * 1024 * 1024:
            st.error("CSV is larger than 10 MB. Please upload a smaller file.")
            st.stop()

        batch = pd.read_csv(uploaded)

        st.markdown("### Dataset Health")
        show_dataset_health(batch)

        required_cols = {
            "MaxBookees", "Hour", "TimeSlot", "Month", "Day",
            "ActivitySiteID", "ActivityDescription", "CurrentPrice"
        }
        missing_cols = required_cols - set(batch.columns)

        if missing_cols:
            st.error("Missing required columns: " + ", ".join(sorted(missing_cols)))
        elif batch.empty:
            st.warning("The uploaded CSV is empty.")
        else:
            @st.cache_data(show_spinner=False)
            def optimize_batch_cached(batch_csv, model_columns):
                batch = pd.read_csv(pd.io.common.StringIO(batch_csv))

                # Same pricing logic as find_optimal_price, but all rows are
                # evaluated together instead of using nested Python loops.
                # Local 5% price guardrail, consistent with the project's controlled
                # dynamic-pricing approach and the existing price sensitivity.
                base_prices = batch["CurrentPrice"].to_numpy(dtype=float)
                n_rows = len(batch)
                step = 5
                low_prices = np.maximum(50, base_prices * 0.95)
                high_prices = base_prices * 1.05
                price_lists = [np.unique(np.append(
                    np.arange(np.floor(lo / step) * step, np.ceil(hi / step) * step + step, step), bp
                )) for lo, hi, bp in zip(low_prices, high_prices, base_prices)]
                n_prices_per_row = np.array([len(x) for x in price_lists], dtype=int)
                prices = np.concatenate(price_lists)

                candidates = pd.DataFrame({
                    "Price (INR)": prices,
                    "MaxBookees": np.repeat(batch["MaxBookees"].to_numpy(), n_prices_per_row),
                    "Hour": np.repeat(batch["Hour"].to_numpy(), n_prices_per_row),
                    "TimeSlot": np.repeat(batch["TimeSlot"].to_numpy(), n_prices_per_row),
                    "Month": np.repeat(batch["Month"].to_numpy(), n_prices_per_row),
                    "Day": np.repeat(batch["Day"].to_numpy(), n_prices_per_row),
                    "ActivitySiteID": np.repeat(batch["ActivitySiteID"].to_numpy(), n_prices_per_row),
                    "ActivityDescription": np.repeat(batch["ActivityDescription"].to_numpy(), n_prices_per_row),
                })

                encoded = pd.get_dummies(
                    candidates,
                    columns=["TimeSlot", "Month", "Day", "ActivitySiteID", "ActivityDescription"]
                ).reindex(columns=list(model_columns), fill_value=0)

                predicted = np.asarray(rf_model.predict(encoded), dtype=float)
                predicted = np.clip(
                    predicted,
                    0,
                    candidates["MaxBookees"].to_numpy(dtype=float)
                )
                revenue = candidates["Price (INR)"].to_numpy() * predicted

                candidate_results = pd.DataFrame({
                    "RowID": np.repeat(np.arange(n_rows), n_prices_per_row),
                    "Price": candidates["Price (INR)"].to_numpy(),
                    "PredictedBookings": predicted,
                    "Revenue": revenue,
                })

                best_idx = candidate_results.groupby("RowID")["Revenue"].idxmax()
                best = candidate_results.loc[best_idx].sort_values("RowID").reset_index(drop=True)

                base_rows = batch.reset_index(drop=True).copy()
                base_candidates = pd.DataFrame({
                    "Price (INR)": base_rows["CurrentPrice"].to_numpy(),
                    "MaxBookees": base_rows["MaxBookees"].to_numpy(),
                    "Hour": base_rows["Hour"].to_numpy(),
                    "TimeSlot": base_rows["TimeSlot"].to_numpy(),
                    "Month": base_rows["Month"].to_numpy(),
                    "Day": base_rows["Day"].to_numpy(),
                    "ActivitySiteID": base_rows["ActivitySiteID"].to_numpy(),
                    "ActivityDescription": base_rows["ActivityDescription"].to_numpy(),
                })

                base_encoded = pd.get_dummies(
                    base_candidates,
                    columns=["TimeSlot", "Month", "Day", "ActivitySiteID", "ActivityDescription"]
                ).reindex(columns=list(model_columns), fill_value=0)

                base_pred = np.asarray(rf_model.predict(base_encoded), dtype=float)
                base_pred = np.clip(
                    base_pred,
                    0,
                    base_rows["MaxBookees"].to_numpy(dtype=float)
                )
                base_rev = base_rows["CurrentPrice"].to_numpy(dtype=float) * base_pred

                result_df = base_rows.copy()
                result_df["RecommendedPrice"] = best["Price"].round().to_numpy()
                result_df["ExpectedBookings"] = best["PredictedBookings"].round().to_numpy()
                result_df["ExpectedRevenue"] = best["Revenue"].round().to_numpy()
                result_df["RevenueUplift%"] = np.where(
                    base_rev > 0,
                    ((best["Revenue"].to_numpy() - base_rev) / base_rev) * 100,
                    0
                ).round(1)

                return result_df

            with st.spinner("Optimizing uploaded classes..."):
                result_df = optimize_batch_cached(
                    uploaded.getvalue().decode("utf-8"),
                    tuple(RF_COLUMNS)
                )

            st.caption(
                "Pipeline: uploaded data → validation → vectorized Random Forest prediction "
                "→ local ±5% candidate-price sweep → revenue optimization."
            )

            st.success(f"Optimized {len(result_df):,} rows.")
            st.dataframe(result_df, use_container_width=True, hide_index=True)
            st.download_button(
                "Download results CSV",
                result_df.to_csv(index=False),
                "optimized_prices.csv",
                "text/csv"
            )

            s1, s2, s3 = st.columns(3)
            s1.metric(
                "Avg Recommended Price",
                f"₹{result_df['RecommendedPrice'].mean():,.0f}"
            )
            s2.metric(
                "Avg Expected Bookings",
                f"{result_df['ExpectedBookings'].mean():,.1f}"
            )
            s3.metric(
                "Total Expected Revenue",
                f"₹{result_df['ExpectedRevenue'].sum():,.0f}"
            )
            st.metric(
                "Average Revenue Uplift Across Uploaded Classes",
                f"{result_df['RevenueUplift%'].mean():+.1f}%"
            )

            # ------------------------------------------------------------
            # LIVE EVALUATION: only possible when uploaded data has the
            # real observed target "Number Booked".
            # ------------------------------------------------------------
            live_eval = evaluate_uploaded_data(batch)

            if live_eval is not None:
                st.markdown("### Live Evaluation on Uploaded Dataset")
                st.caption(
                    "These metrics are calculated from the uploaded rows, "
                    "not from the original training baseline."
                )

                m1, m2, m3 = st.columns(3)
                m1.metric("R²", f"{live_eval['r2']:.4f}")
                m2.metric("RMSE", f"{live_eval['rmse']:.4f}")
                m3.metric("MAE", f"{live_eval['mae']:.4f}")

                live_plot = pd.DataFrame({
                    "Actual": live_eval["actual"],
                    "Predicted": live_eval["predicted"]
                })

                fig_live = px.scatter(
                    live_plot,
                    x="Actual",
                    y="Predicted",
                    title="Actual vs Predicted Bookings — Uploaded Dataset",
                    opacity=0.65,
                    trendline="ols"
                )
                min_v = float(min(live_plot["Actual"].min(), live_plot["Predicted"].min()))
                max_v = float(max(live_plot["Actual"].max(), live_plot["Predicted"].max()))
                fig_live.add_shape(
                    type="line",
                    x0=min_v, y0=min_v, x1=max_v, y1=max_v,
                    line=dict(dash="dash")
                )
                fig_live.update_layout(
                    xaxis_title="Actual Bookings",
                    yaxis_title="Predicted Bookings"
                )
                st.plotly_chart(fig_live, use_container_width=True)

            else:
                st.info(
                    "To update R², RMSE, MAE and Actual-vs-Predicted dynamically, "
                    "the uploaded CSV must contain the real `Number Booked` column. "
                    "The current 8-column optimizer file has no actual target, so "
                    "the app correctly avoids showing a misleading score."
                )

            # Always update the uploaded-data prediction view, even when the
            # real target is not available.
            st.markdown("### Uploaded Dataset — Prediction View")
            pred_view = result_df.copy()
            pred_view["Row"] = np.arange(1, len(pred_view) + 1)

            fig_book = px.line(
                pred_view.head(50),
                x="Row",
                y="ExpectedBookings",
                title="Predicted Bookings — First 50 Uploaded Rows"
            )
            fig_book.update_layout(xaxis_title="Uploaded Row", yaxis_title="Predicted Bookings")
            st.plotly_chart(fig_book, use_container_width=True)

            fig_rev = px.line(
                pred_view.head(50),
                x="Row",
                y="ExpectedRevenue",
                title="Expected Revenue — First 50 Uploaded Rows"
            )
            fig_rev.update_layout(xaxis_title="Uploaded Row", yaxis_title="Expected Revenue (INR)")
            st.plotly_chart(fig_rev, use_container_width=True)


# ====================================================================
# TAB 5: Model Insights — explainability + validation
# ====================================================================
with tab5:
    st.subheader("Model Explainability & Validation — Training Baseline")
 
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
            validation_rows = hist_sample[[
                "Price (INR)", "MaxBookees", "Hour", "TimeSlot", "Month", "Day",
                "ActivitySiteID", "ActivityDescription"
            ]].copy()
            preds = predict_bookings_batch(validation_rows)
            actual = hist_sample["Number Booked"].values.astype(float)
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
    - **Optimal Price Finder**: performs a local candidate-price sweep around the current price,
      using the RF model to maximize `Price × Predicted Bookings` while applying a ±5% practical price guardrail.
    - **Elasticity**: computed directly from the RF model's own demand response around the
      current price, rather than assumed from a rule table.
    - **Batch Optimizer**: applies the same optimization to many classes at once for
      operational, portfolio-level pricing decisions.
    """)
st.caption("⚠️ Predictions are estimates based on historical data (2018 fitness class bookings) — "
           "use as decision support, not absolute truth.")
