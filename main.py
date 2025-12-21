import streamlit as st
from datetime import datetime
import pandas as pd
import requests
import plotly.express as px
import plotly.graph_objects as go
import asyncio
import aiohttp
import time
from multiprocessing import Pool, cpu_count

def process_city(city_df):
    city_df = city_df.sort_values("timestamp")
    city_df["ma"] = city_df["temperature"].rolling(30, min_periods=30).mean()
    city_df["std"] = city_df["temperature"].rolling(30, min_periods=30).std()
    city_df["ma_365"] = city_df["temperature"].rolling(365, min_periods=365).mean()
    return city_df


st.set_page_config(page_title="Temperature Analysis", layout="wide")
st.title("Temperature Analysis")

if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None
if "current_temp" not in st.session_state:
    st.session_state.current_temp = None


uploaded_file = st.file_uploader("Firstly choose a file")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    required_cols = {"city", "timestamp", "temperature", "season"}
    missing = required_cols - set(df.columns)
    if missing:
        st.error(f"Missing columns in uploaded file: {missing}")
        st.stop()

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["city", "timestamp"])

    df["season"] = (
        df["season"]
        .astype(str)
        .str.lower()
        .str.strip()
        .replace({
            "зима": "winter",
            "весна": "spring",
            "лето": "summer",
            "осень": "autumn"
        })
    )

    st.session_state.uploaded_file = df

    st.subheader("Example of Historic Data")
    st.dataframe(df, use_container_width=True)

cities = [
    'Beijing', 'Berlin', 'Cairo', 'Dubai', 'London', 'Los Angeles',
    'Mexico City', 'Moscow', 'Mumbai', 'New York', 'Paris',
    'Rio de Janeiro', 'Singapore', 'Sydney', 'Tokyo'
]

option_city = st.selectbox("Choose a country", cities)
api_key = st.text_input("Enter your API key", type="password")


def get_weather_sync(city, api_key):
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {"q": city, "appid": api_key, "units": "metric"}
    r = requests.get(url, params=params, timeout=20)
    return r.json()


async def get_weather_async(city, api_key):
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {"q": city, "appid": api_key, "units": "metric"}

    async with aiohttp.ClientSession() as session:
        async with session.get(url, params=params) as resp:
            return await resp.json()


if api_key:
    data = get_weather_sync(option_city, api_key)

    if data.get("cod") == 401:
        st.error(data.get("message"))
        st.stop()

    st.session_state.current_temp = data["main"]["temp"]
    st.success(
        f"Current temperature in {option_city}: "
        f"{st.session_state.current_temp:.1f} °C"
    )


if api_key:
    with st.expander("Async vs Sync API experiment"):
        start = time.time()
        get_weather_sync(option_city, api_key)
        sync_time = time.time() - start

        start = time.time()
        try:
            asyncio.run(get_weather_async(option_city, api_key))
        except RuntimeError:
            loop = asyncio.new_event_loop()
            loop.run_until_complete(get_weather_async(option_city, api_key))
            loop.close()
        async_time = time.time() - start

        st.write(f"Sync request time: {sync_time:.4f} sec")
        st.write(f"Async request time: {async_time:.4f} sec")

def get_current_season(date=None):
    if date is None:
        date = datetime.now()
    m = date.month
    if m in [12, 1, 2]:
        return "winter"
    elif m in [3, 4, 5]:
        return "spring"
    elif m in [6, 7, 8]:
        return "summer"
    else:
        return "autumn"


if option_city and api_key:

    if st.session_state.uploaded_file is None:
        st.warning("Please upload historical temperature data first.")
        st.stop()

    df = st.session_state.uploaded_file

#доп - выбор окна для ma()
    window_size = st.slider(
        "Moving average window (days)",
        min_value=30,
        max_value=90,
        step=10,
        value=30
    )

    #multiprocessing experiment
    if st.button("Run multiprocessing experiment"): #замеряется время подсчета ma(30) и ma(365) - самая первая функция в коде
        start_seq = time.time()
        for _, g in df.groupby("city"):
            process_city(g)
        seq_time = time.time() - start_seq

        start_par = time.time()
        city_parts = [g for _, g in df.groupby("city")]

        with Pool(processes=min(cpu_count(), len(city_parts))) as pool:
            pool.map(process_city, city_parts)

        par_time = time.time() - start_par

        st.write(f"Sequential time: {seq_time:.4f} sec")
        st.write(f"Parallel time: {par_time:.4f} sec")
        st.caption("Multiprocessing is useful only for large datasets.")

    g = df.groupby("city")["temperature"]
    df["ma"] = g.rolling(window_size, min_periods=window_size).mean().reset_index(level=0, drop=True)
    df["std"] = g.rolling(window_size, min_periods=window_size).std().reset_index(level=0, drop=True)
    df["ma_365"] = g.rolling(365, min_periods=365).mean().reset_index(level=0, drop=True)


    fig_ma = px.line(
        df[df.city == option_city],
        x="timestamp",
        y="ma",
        title=f"{window_size}-day Moving Average Temperature in {option_city}"
    )
    st.plotly_chart(fig_ma, use_container_width=True)

    df["is_anomaly"] = (
        (df["temperature"] > df["ma"] + 2 * df["std"]) |
        (df["temperature"] < df["ma"] - 2 * df["std"])
    )

    city_df = df[df.city == option_city]
    anomalies = city_df[city_df.is_anomaly]

    fig_anom = go.Figure()
    fig_anom.add_trace(go.Scatter(
        x=city_df["timestamp"],
        y=city_df["temperature"],
        mode="markers",
        name="Temperature"
    ))
    fig_anom.add_trace(go.Scatter(
        x=anomalies["timestamp"],
        y=anomalies["temperature"],
        mode="markers",
        marker=dict(color="red"),
        name="Anomalies"
    ))
    fig_anom.update_layout(
        title=f"Temperature with Anomalies in {option_city}",
        hovermode="x unified"
    )
    st.plotly_chart(fig_anom, use_container_width=True)


    fig_long = px.line(
        city_df.dropna(),
        x="timestamp",
        y="ma_365",
        title=f"365-day Moving Average in {option_city}"
    )
    st.plotly_chart(fig_long, use_container_width=True)

    st.subheader(f"Seasonal Statistics for {option_city}")

    season_stats = (
        city_df.groupby("season")["temperature"]
        .agg(["mean", "std"])
        .reset_index()
    )
    st.dataframe(season_stats, use_container_width=True)

    fig_season = go.Figure()
    fig_season.add_trace(go.Bar(
        x=season_stats["season"],
        y=season_stats["mean"],
        error_y=dict(type="data", array=season_stats["std"]),
        name="Mean ± Std"
    ))
    fig_season.update_layout(
        title=f"Seasonal Temperature Profile for {option_city}",
        xaxis_title="Season",
        yaxis_title="Temperature (°C)"
    )
    st.plotly_chart(fig_season, use_container_width=True)

    st.subheader("Number of anomalies by season")

    anomalies_by_season = (
        city_df[city_df.is_anomaly]
        .groupby("season")
        .size()
        .reset_index(name="anomaly_count")
    )
# доп - глянем сколько аномалий по сезонам в целом
    all_seasons = pd.DataFrame({"season": ["winter", "spring", "summer", "autumn"]})
    anomalies_by_season = all_seasons.merge(
        anomalies_by_season, on="season", how="left"
    ).fillna(0)

    st.dataframe(anomalies_by_season, use_container_width=True)

    fig_anom_season = px.bar(
        anomalies_by_season,
        x="season",
        y="anomaly_count",
        title=f"Number of anomaly days by season in {option_city}"
    )
    st.plotly_chart(fig_anom_season, use_container_width=True)

#доп фича - скачать все аномалии для выбранного города
    if not anomalies.empty:
        st.download_button(
            f"Download all historical anomalies for {option_city} CSV",
            anomalies.to_csv(index=False),
            file_name=f"{option_city}_anomalies.csv",
            mime="text/csv"
        )

    current_season = get_current_season()
    season_row = season_stats[season_stats.season == current_season]

    lower = season_row["mean"].iloc[0] - 2 * season_row["std"].iloc[0]
    upper = season_row["mean"].iloc[0] + 2 * season_row["std"].iloc[0]

    st.header("Current Temperature Analysis")
    st.metric("Current Temperature", f"{st.session_state.current_temp:.1f} °C")
    st.write(f"Season: **{current_season}**")
    st.write(f"Normal range: **{lower:.1f} – {upper:.1f} °C**")

    if lower <= st.session_state.current_temp <= upper:
        st.success("Current temperature is within the seasonal norm")
    else:
        st.error("Current temperature is anomalous for this season")


if __name__ == "__main__":
    pass
