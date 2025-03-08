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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ===============================
# PART 1: Bottom-Up Market Sizing for IPF
# ===============================

# Dummy data for approved drugs for IPF (dummy revenues in USD)
drug_data = [
    {"DrugName": "DrugX", "Indication": "IPF", "IsApproved": True,  "Revenue_2021": 0.8e9, "Revenue_2022": 0.85e9},
    {"DrugName": "DrugY", "Indication": "IPF", "IsApproved": True,  "Revenue_2021": 1.0e9, "Revenue_2022": 1.1e9},
    {"DrugName": "DrugZ", "Indication": "IPF", "IsApproved": True,  "Revenue_2021": 0.6e9, "Revenue_2022": 0.65e9},
    {"DrugName": "OtherDrug", "Indication": "Other", "IsApproved": True, "Revenue_2021": 0.3e9, "Revenue_2022": 0.35e9}
]
df_drugs = pd.DataFrame(drug_data)

# Filter for IPF-approved drugs
target_indication = "IPF"
approved_drugs = df_drugs[(df_drugs["Indication"] == target_indication) & (df_drugs["IsApproved"] == True)].copy()

# Assume 2022 revenue is fully attributable to IPF
approved_drugs["Total_Revenue_2022"] = approved_drugs["Revenue_2022"]

# Bottom-Up Market Sizing for IPF (2022)
bottom_up_market_size = approved_drugs["Total_Revenue_2022"].sum()
print("Bottom-Up Market Size for IPF (2022): ${:.2f} Billion".format(bottom_up_market_size / 1e9))

# -------------------------------
# Determine baseline competitor market share fractions based on 2022 revenue
# -------------------------------
# Calculate each competitor's share (of the IPF market) in 2022:
approved_drugs["BaselineFraction"] = approved_drugs["Total_Revenue_2022"] / bottom_up_market_size
# For reference, display the baseline fractions:
print("\nBaseline competitor fractions (2022):")
print(approved_drugs[["DrugName", "BaselineFraction"]])

# ===============================
# PART 2: 10-Year Forecast for BMS-Squib with Competitor Evolution
# ===============================

# Forecast years: from 2022 (baseline) to 2032
forecast_years = np.arange(2022, 2032 + 1)

# Assume overall IPF market grows at 5% per year (starting from 2022 baseline)
market_growth_rate = 0.05
market_size_forecast = [bottom_up_market_size * (1 + market_growth_rate) ** (year - 2022) for year in forecast_years]

# Define new product launch details:
# BMS-Squib launches in 2023 with 1% market share, growing linearly to 10% by 2032.
bms_squib_shares = []
for year in forecast_years:
    if year < 2023:
        bms_squib_shares.append(0.0)  # Not launched yet
    else:
        # Linear growth from 1% (2023) to 10% (2032)
        bms_share = 0.01 + (0.10 - 0.01) * ((year - 2023) / (2032 - 2023))
        bms_squib_shares.append(bms_share)

# For each forecast year, the competitor total share = 100% - BMS-Squib share.
competitor_total_shares = [1 - s for s in bms_squib_shares]

# For each competitor, assume their share is proportional to their baseline fraction.
# We'll store forecasted market shares and revenues in dictionaries.
competitor_names = approved_drugs["DrugName"].tolist()
competitor_forecast_shares = {name: [] for name in competitor_names}
competitor_forecast_revenues = {name: [] for name in competitor_names}

bms_squib_revenues = []

# Build forecast for each year:
for i, year in enumerate(forecast_years):
    # Total market size for this year:
    total_market = market_size_forecast[i]

    # BMS-Squib revenue:
    bms_share = bms_squib_shares[i]
    bms_revenue = total_market * bms_share
    bms_squib_revenues.append(bms_revenue)

    # Competitor total share:
    comp_total_share = competitor_total_shares[i]

    # For each competitor, assign share = comp_total_share * (baseline fraction)
    for idx, name in enumerate(competitor_names):
        baseline_frac = approved_drugs.loc[approved_drugs["DrugName"] == name, "BaselineFraction"].values[0]
        comp_share = comp_total_share * baseline_frac
        competitor_forecast_shares[name].append(comp_share)

        # Revenue for this competitor:
        comp_revenue = total_market * comp_share
        competitor_forecast_revenues[name].append(comp_revenue)

# Build a forecast DataFrame
forecast_df = pd.DataFrame({
    "Year": forecast_years,
    "Market_Size_USD": market_size_forecast,
    "BMS_Squib_Market_Share": bms_squib_shares,
    "BMS_Squib_Revenue_USD": bms_squib_revenues,
})

# Add competitor columns
for name in competitor_names:
    forecast_df[f"{name}_Market_Share"] = competitor_forecast_shares[name]
    forecast_df[f"{name}_Revenue_USD"] = competitor_forecast_revenues[name]

# Display the forecast table
pd.options.display.float_format = '{:,.2f}'.format
print("\n10-Year Forecast for IPF Market with BMS-Squib & Competitors:")
print(forecast_df)

# ===============================
# PART 3: Visualizations
# ===============================

# Plot overall market size forecast
plt.figure(figsize=(10, 6))
plt.plot(forecast_df["Year"], forecast_df["Market_Size_USD"] / 1e9, marker="o", color="blue")
plt.title("Forecasted IPF Market Size (2022-2032)")
plt.xlabel("Year")
plt.ylabel("Market Size (Billion USD)")
plt.grid(True)
plt.show()

# Plot BMS-Squib market share evolution
plt.figure(figsize=(10, 6))
plt.plot(forecast_df["Year"], np.array(forecast_df["BMS_Squib_Market_Share"]) * 100, marker="o", label="BMS-Squib", color="red")
plt.title("BMS-Squib Market Share Evolution")
plt.xlabel("Year")
plt.ylabel("Market Share (%)")
plt.legend()
plt.grid(True)
plt.show()

# Plot competitor market share evolution (stacked)
plt.figure(figsize=(10, 6))
for name in competitor_names:
    plt.plot(forecast_df["Year"], np.array(forecast_df[f"{name}_Market_Share"]) * 100, marker="o", label=name)
plt.title("Competitor Market Share Evolution")
plt.xlabel("Year")
plt.ylabel("Market Share (%)")
plt.legend()
plt.grid(True)
plt.show()

# Plot revenue forecast for BMS-Squib and competitors
plt.figure(figsize=(10, 6))
plt.plot(forecast_df["Year"], np.array(forecast_df["BMS_Squib_Revenue_USD"]) / 1e9, marker="o", label="BMS-Squib", color="red")
for name in competitor_names:
    plt.plot(forecast_df["Year"], np.array(forecast_df[f"{name}_Revenue_USD"]) / 1e9, marker="o", label=name)
plt.title("Revenue Forecast (USD Billion)")
plt.xlabel("Year")
plt.ylabel("Revenue (Billion USD)")
plt.legend()
plt.grid(True)
plt.show()
