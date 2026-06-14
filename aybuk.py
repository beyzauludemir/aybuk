
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy import stats
import math

st.set_page_config(page_title="Steel Inventory Intelligence", layout="wide")

ORDERING_COST = 2792
HOLDING_COST_PCT = 0.25

st.title("Steel Inventory Intelligence - Diesel (Motorin)")

uploaded_file = st.sidebar.file_uploader("Upload motorin tüketim.xlsx", type=["xlsx"])
service_level = st.sidebar.selectbox("Service Level", [0.90, 0.95, 0.97, 0.99], index=1)
forecast_horizon = st.sidebar.slider("Forecast Horizon (Days)", 30, 365, 90)

tabs = st.tabs([
    "ABC-XYZ Analysis",
    "Forecast",
    "Inventory Optimization",
    "Visualization",
    "Executive Summary"
])

if uploaded_file is not None:

    df = pd.read_excel(uploaded_file)
    df["Tarih"] = pd.to_datetime(df["Tarih"])
    df = df.sort_values("Tarih")

    # Simple forecast MVP
    df["Forecast_Base"] = df["Tüketim"].rolling(7, min_periods=1).mean()

    last_date = df["Tarih"].max()
    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=forecast_horizon,
        freq="D"
    )

    forecast_value = float(df["Tüketim"].tail(30).mean())

    forecast_df = pd.DataFrame({
        "Date": future_dates,
        "Forecast": forecast_value
    })

    avg_daily_demand = forecast_df["Forecast"].mean()
    annual_demand = avg_daily_demand * 365
    std_demand = max(df["Tüketim"].std(), 1)

    lead_time = 3
    unit_price = 45

    holding_cost = HOLDING_COST_PCT * unit_price

    eoq = math.sqrt((2 * annual_demand * ORDERING_COST) / holding_cost)

    z = stats.norm.ppf(service_level)
    safety_stock = z * std_demand * np.sqrt(lead_time)

    rop = avg_daily_demand * lead_time + safety_stock

    with tabs[0]:
        st.subheader("ABC-XYZ Analysis")

        annual_value = df["Tüketim"].sum() * unit_price

        cv = (df["Tüketim"].std() / df["Tüketim"].mean()) * 100

        abc_class = "A"

        if cv < 10:
            xyz_class = "X"
        elif cv < 25:
            xyz_class = "Y"
        else:
            xyz_class = "Z"

        st.metric("ABC Class", abc_class)
        st.metric("XYZ Class", xyz_class)
        st.metric("Annual Consumption Value", f"{annual_value:,.0f}")

    with tabs[1]:
        st.subheader("Forecast")

        fig = px.line(df, x="Tarih", y="Tüketim")
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(forecast_df)

    with tabs[2]:
        st.subheader("Inventory Optimization")

        c1, c2, c3 = st.columns(3)

        c1.metric("EOQ", f"{eoq:,.0f}")
        c2.metric("ROP", f"{rop:,.0f}")
        c3.metric("Safety Stock", f"{safety_stock:,.0f}")

    with tabs[3]:
        st.subheader("Visualization")

        combined = pd.concat([
            pd.DataFrame({
                "Date": df["Tarih"],
                "Value": df["Tüketim"],
                "Type": "Historical"
            }),
            pd.DataFrame({
                "Date": forecast_df["Date"],
                "Value": forecast_df["Forecast"],
                "Type": "Forecast"
            })
        ])

        fig2 = px.line(combined, x="Date", y="Value", color="Type")
        st.plotly_chart(fig2, use_container_width=True)

    with tabs[4]:
        st.subheader("Executive Summary")

        st.markdown(f"""
        ### Material: Diesel

        - ABC XYZ Class: **{abc_class}{xyz_class}**
        - Forecast Horizon: **{forecast_horizon} days**
        - Average Forecast Demand: **{avg_daily_demand:,.0f}**
        - EOQ: **{eoq:,.0f}**
        - ROP: **{rop:,.0f}**
        - Safety Stock: **{safety_stock:,.0f}**
        - Recommended Policy: **Continuous Review**
        """)

else:
    st.info("Please upload motorin tüketim.xlsx")
