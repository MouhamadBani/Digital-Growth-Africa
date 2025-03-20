import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import datetime

# Apply Dark Theme
st.markdown("""
    <style>
        body {background-color: #121212; color: white;}
        .block-container {padding-top: 20px;}
        .stButton>button {background-color: #ff4b4b; color: white;}
        .stSelectbox div[data-baseweb="select"] {background-color: #292929 !important; color: white !important;}
    </style>
    """, unsafe_allow_html=True)

# API Endpoints
WORLD_BANK_API = "https://api.worldbank.org/v2/country/{}/indicator/{}?format=json"

# List of all African Countries with ISO3 codes
africa_countries = {
    "Algeria": "DZA", "Angola": "AGO", "Benin": "BEN", "Botswana": "BWA", "Burkina Faso": "BFA",
    "Burundi": "BDI", "Cabo Verde": "CPV", "Cameroon": "CMR", "Central African Republic": "CAF",
    "Chad": "TCD", "Comoros": "COM", "Democratic Republic of the Congo": "COD", "Republic of the Congo": "COG",
    "Djibouti": "DJI", "Egypt": "EGY", "Equatorial Guinea": "GNQ", "Eritrea": "ERI", "Eswatini": "SWZ",
    "Ethiopia": "ETH", "Gabon": "GAB", "Gambia": "GMB", "Ghana": "GHA", "Guinea": "GIN",
    "Guinea-Bissau": "GNB", "Ivory Coast": "CIV", "Kenya": "KEN", "Lesotho": "LSO", "Liberia": "LBR",
    "Libya": "LBY", "Madagascar": "MDG", "Malawi": "MWI", "Mali": "MLI", "Mauritania": "MRT",
    "Mauritius": "MUS", "Morocco": "MAR", "Mozambique": "MOZ", "Namibia": "NAM", "Niger": "NER",
    "Nigeria": "NGA", "Rwanda": "RWA", "São Tomé and Príncipe": "STP", "Senegal": "SEN", "Seychelles": "SYC",
    "Sierra Leone": "SLE", "Somalia": "SOM", "South Africa": "ZAF", "South Sudan": "SSD", "Sudan": "SDN",
    "Tanzania": "TZA", "Togo": "TGO", "Tunisia": "TUN", "Uganda": "UGA", "Zambia": "ZMB", "Zimbabwe": "ZWE"
}

# Streamlit UI
st.markdown("<h1 style='text-align: center; color: red;'> Africa Digital payments Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Tracking digital payments, fintech growth, and crypto adoption.</p>", unsafe_allow_html=True)

# Sidebar - Country Selection
country = st.sidebar.selectbox("🌍 **Select a Country**", list(africa_countries.keys()))
iso_code = africa_countries[country]

# Function to Fetch World Bank Data
@st.cache_data
def fetch_world_bank_data(iso_code, indicator):
    """Fetches data from the World Bank API and returns a cleaned dataframe."""
    url = WORLD_BANK_API.format(iso_code, indicator)
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if isinstance(data, list) and len(data) > 1:
            df = pd.DataFrame(data[1]).dropna(subset=['value'])
            df["date"] = df["date"].astype(int)  # Ensure year is treated as integer
            df = df.sort_values(by="date")
            return df
    return pd.DataFrame()

# Fetch Digital Payments Data for all countries
@st.cache_data
def fetch_africa_fintech_data(indicator):
    """Fetches fintech adoption data for all African countries."""
    fintech_data = {}
    for country, iso in africa_countries.items():
        df = fetch_world_bank_data(iso, indicator)
        if not df.empty:
            latest_year = df["date"].max()
            latest_value = df[df["date"] == latest_year]["value"].values[0]
            fintech_data[iso] = latest_value
    return fintech_data

# Fetch data for selected country
digital_payments_data = fetch_world_bank_data(iso_code, "IT.NET.USER.ZS")

# Fetch fintech adoption rates for the entire map
africa_fintech_data = fetch_africa_fintech_data("IT.NET.USER.ZS")

# Display Metrics
col1, col2, col3 = st.columns(3)
if not digital_payments_data.empty:
    latest_year = digital_payments_data["date"].max()
    latest_value = digital_payments_data[digital_payments_data["date"] == latest_year]["value"].values[0]

    col1.metric("📈 Digital Payments Usage", f"{latest_value:.2f}%")
    col2.metric("📅 Latest Data Year", f"{latest_year}")
    col3.metric("🌍 Country", f"{country}")

# AI Prediction Model for Fintech Growth Over 5-10 Years
def predict_digital_banking_growth(data, years=10):
    """Uses Linear Regression to predict fintech adoption growth."""
    if not data.empty:
        data["date"] = pd.to_datetime(data["date"], format='%Y')
        data["year"] = data["date"].dt.year
        X = np.array(data["year"]).reshape(-1, 1)
        y = np.array(data["value"]).reshape(-1, 1)
        model = LinearRegression()
        model.fit(X, y)
        future_years = np.array([[datetime.now().year + i] for i in range(1, years + 1)])
        predictions = model.predict(future_years)
        return dict(zip(future_years.flatten(), predictions.flatten()))
    return {}

# AI Forecast for 5-10 Years
future_growth = predict_digital_banking_growth(digital_payments_data, years=10)
if future_growth:
    forecast_text = "\n".join([f"📊 {year}: {growth:.2f}%" for year, growth in future_growth.items()])
    st.info(f"### Projection for Digital Payment Growth\n{forecast_text}")

# Visualize Data as a Choropleth Map of Africa
st.subheader("Digital Payments Adoption in Africa")
df_map = pd.DataFrame(africa_fintech_data.items(), columns=["iso_code", "fintech_adoption"])

if not df_map.empty:
    fig_map = px.choropleth(df_map, locations="iso_code",
                             locationmode="ISO-3",
                             color="fintech_adoption",
                             title="Fintech Adoption Rate in Africa",
                             projection="natural earth",
                             scope="africa",
                             color_continuous_scale="Viridis",  
                             labels={"fintech_adoption": "Fintech Adoption (%)"})  
    st.plotly_chart(fig_map, use_container_width=True)
else:
    st.warning("No data available for fintech adoption map.")

# Line Chart for Digital Payments Growth Over Time
st.subheader(f"Digital Payments Growth in {country}")
if not digital_payments_data.empty:
    fig2 = px.line(digital_payments_data, x="date", y="value",
                   title=f"Internet & Digital Payments Growth in {country}",
                   labels={"value": "Internet Users (% of Population)", "date": "Year"})
    st.plotly_chart(fig2)
else:
    st.warning("No digital payments data available.")

# Footer
st.markdown("<p style='text-align: center;'>Data Source: World Bank API | Developed by Mouhamad Bani in Streamlit</p>", unsafe_allow_html=True)
