import streamlit as st
from datetime import datetime
import pandas as pd
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import asyncio
import aiohttp
import time
from multiprocessing import Pool, cpu_count

# 4c8fa7d9b32c88b547108c4e9e67ed7c
st.set_page_config(page_title="Temperature Analysis", layout="wide")

st.title('Temperature Analysis')

if "visibility" not in st.session_state:
    st.session_state.visibility = "visible"
    st.session_state.api_key = None
    st.session_state.city = None
    st.session_state.current_temp = None
    st.session_state.uploaded_file = None



uploaded_file = st.file_uploader("Choose a file")

if uploaded_file is not None:
  
  df = pd.read_csv(uploaded_file)
  st.session_state.uploaded_file = df
  st.text('Example of Historic Data')
  required_cols = {"city", "timestamp", "temperature", "season"}
  missing = required_cols - set(df.columns)

  if missing:
        st.error(f"Missing columns in uploaded file: {missing}")
        st.stop()

  st.write(df)

cities = ['Beijing', 'Berlin', 'Cairo', 'Dubai', 'London', 'Los Angeles',
       'Mexico City', 'Moscow', 'Mumbai', 'New York', 'Paris',
       'Rio de Janeiro', 'Singapore', 'Sydney', 'Tokyo']

option_city = st.selectbox(
    "Choose a country",
    cities,
)


api_key = st.text_input(
"Enter your API key ",
type="password",
label_visibility=st.session_state.visibility)


def get_weather_sync(city, api_key):
    """
    синхронный способ
    """
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        "q": city,
        "appid": api_key,
        "units": "metric"
    }
    r = requests.get(url, params=params)
    return r.json()


if api_key:
    data = get_weather_sync(option_city, api_key)

    if data.get("cod") == 401:
        st.error(data.get("message"))
        st.stop()

    st.session_state.current_temp = data["main"]["temp"]
    st.success(f"Current temperature in {option_city}: {st.session_state.current_temp} °C")



async def get_weather_async(city, api_key):
    """
    асинхронный реквест
    """
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        "q": city,
        "appid": api_key,
        "units": "metric"
    }

    async with aiohttp.ClientSession() as session:
        async with session.get(url, params=params) as resp:
            return await resp.json()


if api_key:
    with st.expander("Async vs Sync API experiment"):
        start = time.time()
        _ = get_weather_sync(option_city, api_key)
        sync_time = time.time() - start

        start = time.time()
        asyncio.run(get_weather_async(option_city, api_key))
        async_time = time.time() - start

        st.write(f"Sync request time: {sync_time:.4f} sec")
        st.write(f"Async request time: {async_time:.4f} sec")

        st.caption(
            "Async approach is faster with API-request"
        )
def get_current_season(date=None):
    if date is None:
        date = datetime.now()

    month = date.month

    if month in [12, 1, 2]:
        return 'winter'
    elif month in [3, 4, 5]:
        return 'spring'
    elif month in [6, 7, 8]:
        return 'summer'
    else:
        return 'autumn'

if option_city and api_key:
    # df = pd.read_csv(uploaded_file)
    # df = pd.read_csv('temperature_data.csv')
    df = st.session_state.uploaded_file 


    g = df.groupby('city')['temperature']
    df['ma_30'] = g.rolling(30, min_periods=30).mean().reset_index(level=0, drop=True)
    df['std_30'] = g.rolling(30, min_periods=30).std().reset_index(level=0, drop=True)
    df['ma_365'] = (
    g.rolling(365, min_periods=365)
     .mean()
     .reset_index(level=0, drop=True)
)

    fig = px.line(df[df.city == option_city], x = 'timestamp', y = 'ma_30', color_discrete_sequence=["#0514C0"], )
    fig.update_layout(title=f'30-days Moving Average Temperature in {option_city}', xaxis_title='Date', yaxis_title='Temperature')
    st.plotly_chart(fig)
    df['is_anomaly'] = (
    (df['temperature'] > df['ma_30'] + 2 * df['std_30']) |
    (df['temperature'] < df['ma_30'] - 2 * df['std_30'])
        )

    city_df = df[df.city == option_city]
    anomalies = city_df[city_df.is_anomaly]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=city_df['timestamp'],
            y=city_df['temperature'].dropna(),
            mode='markers',
            name='Real temperature',
            # line=dict(color='steelblue')
            marker=dict(color='blue', size=8)

        )
    )

    fig.add_trace(
        go.Scatter(
            x=anomalies['timestamp'],
            y=anomalies['temperature'],
            mode='markers',
            name='Anomalies',
            marker=dict(color='red', size=8)
        )
    )

    fig.update_layout(
        title=f'Real Temperature with Anomalies in {option_city}',
        xaxis_title='Date',
        yaxis_title='Temperature',
        hovermode='x unified',
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)
    st.subheader("Long-term Temperature Trend")

    city_df = df[df.city == option_city]

    fig_long = px.line(
        city_df.dropna(),
        x='timestamp',
        y='ma_365',
        color_discrete_sequence=["#0B7A75"]
    )

    fig_long.update_layout(
        title=f'Long-term Temperature Trend in {option_city} (365-day MA)',
        xaxis_title='Date',
        yaxis_title='Temperature'
    )

    st.plotly_chart(fig_long, use_container_width=True)
    current_season = get_current_season()
    season_stats = (
            df[df.city == option_city]
            .groupby( ['city','season'])['temperature']
            .agg(['mean', 'std'])
        ).reset_index()

    st.subheader("Seasonal Profiles")

    st.write(f'**Seasonal Statistics for {option_city}**')
    st.write(season_stats)

    try:

        season_mean = season_stats[season_stats.season ==current_season ][ 'mean']
        season_std =season_stats[season_stats.season ==current_season ][ 'std']

    except KeyError:
        st.warning(
            f"No historical data for city {option_city} "
            f"and season **{current_season}**"
        )
        st.stop()

    fig_season = go.Figure()

    fig_season.add_trace(
        go.Bar(
            x=season_stats["season"],
            y=season_stats["mean"],
            error_y=dict(type="data", array=season_stats["std"]),
            name="Mean ± std"
        )
    )

    fig_season.update_layout(
        title=f"Seasonal Temperature Profile for {option_city}",
        xaxis_title="Season",
        yaxis_title="Temperature (°C)"
    )

    st.plotly_chart(fig_season, use_container_width=True)


    lower_bound = season_mean - 2 * season_std
    upper_bound = season_mean + 2 * season_std

    if st.session_state.current_temp is None:
        st.warning("Current temperature data is not available. Please enter a valid API key.")
        st.stop()


    is_normal = lower_bound.iloc[0] <= st.session_state.current_temp <= upper_bound.iloc[0]
    st.header("Current Temperature Analysis")

    st.metric(
        label="Current Temperature",
        value=f"{st.session_state.current_temp:.1f} °C"
    )

    st.write(f"**City:** {option_city}")
    st.write(f"**Season:** {current_season.capitalize()}")

    st.write(
        f"**Normal range for this season:** "
        f"{lower_bound.iloc[0]:.1f} °C – {upper_bound.iloc[0]:.1f} °C"
    )

    if is_normal:
        st.success("Current temperature is within the seasonal norm")
    else:
        st.error("Current temperature is anomalous for this season")





