import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Title
st.title("Pharma Market Forecasting App")

# Upload Data
st.sidebar.header("Upload Sales Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Raw Data", df.head())

    # Forecasting
    st.sidebar.header("Forecast Settings")
    periods = st.sidebar.slider("Forecast Periods (Months)", 1, 36, 12)

    # Fit Exponential Smoothing Model
    model = ExponentialSmoothing(df["Sales"], trend="add", seasonal="add", seasonal_periods=12)
    fitted_model = model.fit()
    forecast = fitted_model.forecast(periods)

    # Plot Forecast
    st.write("### Forecasted Sales")
    plt.figure(figsize=(10,5))
    plt.plot(df["Sales"], label="Historical Sales", marker="o")
    plt.plot(range(len(df), len(df) + periods), forecast, label="Forecast", marker="o", linestyle="dashed")
    plt.legend()
    st.pyplot(plt)

    # Market Share Trends
    st.write("### Market Share Trends")
    df["Market Share"] = df["Sales"] / df["Sales"].sum()
    st.line_chart(df["Market Share"])
