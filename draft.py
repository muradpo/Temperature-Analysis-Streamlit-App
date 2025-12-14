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


st.set_page_config(page_title="Temperature Analysis", layout="wide")
st.title("Temperature Analysis")


# =========================
# SESSION STATE
# =========================
if "current_temp" not in st.session_state:
    st.session_state.current_temp = None


# =========================
# FILE UPLOAD
# =========================
uploaded_file = st.file_uploader("Choose a file")

if uploaded_file is None:
    st.stop()

df = pd.read_csv(uploaded_file)

required_cols = {"city", "timestamp", "temperature", "season"}
missing = required_cols - set(df.columns)
if missing:
    st.error(f"Missing columns: {missing}")
    st.stop()

df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values(["city", "timestamp"])

cities = sorted(df["city"].unique())
option_city = st.selectbox("Choose a city", cities)


# =========================
# API KEY
# =========================
api_key = st.text_input("Enter your API key", type="password")


# =========================
# SYNC API
# =========================
def get_weather_sync(city, api_key):
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {"q": city, "appid": api_key, "units": "metric"}
    r = requests.get(url, params=params)
    return r.json()


if api_key:
    data = get_weather_sync(option_city, api_key)
    if data.get("cod") == 401:
        st.error(data.get("message"))
        st.stop()
    st.session_state.current_temp = data["main"]["temp"]
    st.success(f"Current temperature in {option_city}: {st.session_state.current_temp} °C")


# =========================
# ASYNC API
# =========================
async def get_weather_async(city, api_key):
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {"q": city, "appid": api_key, "units": "metric"}
    async with aiohttp.ClientSession() as session:
        async with session.get(url, params=params) as resp:
            return await resp.json()


# =========================
# ASYNC VS SYNC EXPERIMENT
# =========================
if api_key:
    with st.expander("Async vs Sync API experiment"):
        if st.button("Run async vs sync experiment"):
            start = time.time()
            get_weather_sync(option_city, api_key)
            sync_time = time.time() - start

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            start = time.time()
            loop.run_until_complete(get_weather_async(option_city, api_key))
            async_time = time.time() - start
            loop.close()

            st.write(f"Sync request time: {sync_time:.4f} sec")
            st.write(f"Async request time: {async_time:.4f} sec")

            st.caption(
                "Асинхронный подход эффективен при множественных I/O запросах. "
                "Для одного города разница минимальна."
            )


# =========================
# MULTIPROCESSING EXPERIMENT
# =========================
def rolling_worker(city_df):
    """
    Воркер-функция для multiprocessing.

    Здесь выполняется CPU-bound операция:
    расчет скользящего среднего и стандартного отклонения
    для одного города.

    Каждому процессу передается отдельный DataFrame,
    что позволяет распараллелить вычисления.
    """
    city_df = city_df.copy()
    city_df["ma_30"] = city_df["temperature"].rolling(30).mean()
    city_df["std_30"] = city_df["temperature"].rolling(30).std()
    return city_df


with st.expander("Multiprocessing experiment (rolling statistics)"):
    st.write(
        "Эксперимент: сравнение последовательного и параллельного "
        "расчета rolling-статистик по городам."
    )

    if st.button("Run multiprocessing experiment"):
        city_groups = [g for _, g in df.groupby("city")]

        # последовательный расчет
        start = time.time()
        _ = [rolling_worker(g) for g in city_groups]
        seq_time = time.time() - start

        # параллельный расчет
        start = time.time()
        with Pool(processes=cpu_count()) as pool:
            _ = pool.map(rolling_worker, city_groups)
        mp_time = time.time() - start

        st.write(f"Sequential time: {seq_time:.4f} sec")
        st.write(f"Multiprocessing time: {mp_time:.4f} sec")

        st.caption(
            "Multiprocessing ускоряет CPU-bound вычисления "
            "при большом объеме данных. "
            "В основном приложении не используется из-за особенностей Streamlit."
        )


# =========================
# MAIN ANALYSIS
# =========================
g = df.groupby("city")["temperature"]

df["ma_30"] = g.rolling(30, min_periods=30).mean().reset_index(level=0, drop=True)
df["std_30"] = g.rolling(30, min_periods=30).std().reset_index(level=0, drop=True)
df["ma_365"] = g.rolling(365, min_periods=365).mean().reset_index(level=0, drop=True)

df["is_anomaly"] = (
    (df["temperature"] > df["ma_30"] + 2 * df["std_30"]) |
    (df["temperature"] < df["ma_30"] - 2 * df["std_30"])
)

city_df = df[df.city == option_city]
anomalies = city_df[city_df.is_anomaly]


fig = go.Figure()
fig.add_trace(go.Scatter(
    x=city_df["timestamp"],
    y=city_df["temperature"],
    mode="lines",
    name="Temperature"
))
fig.add_trace(go.Scatter(
    x=anomalies["timestamp"],
    y=anomalies["temperature"],
    mode="markers",
    name="Anomalies",
    marker=dict(color="red")
))
st.plotly_chart(fig, use_container_width=True)


fig_long = px.line(
    city_df.dropna(subset=["ma_365"]),
    x="timestamp",
    y="ma_365",
    title="Long-term trend (365-day MA)"
)
st.plotly_chart(fig_long, use_container_width=True)


# =========================
# SEASON CHECK
# =========================
def get_current_season(date=None):
    if date is None:
        date = datetime.now()
    m = date.month
    if m in [12, 1, 2]:
        return "winter"
    if m in [3, 4, 5]:
        return "spring"
    if m in [6, 7, 8]:
        return "summer"
    return "autumn"


season_stats = (
    city_df
    .groupby("season")["temperature"]
    .agg(["mean", "std"])
    .reset_index()
)

current_season = get_current_season()
row = season_stats[season_stats.season == current_season]

lower = row["mean"].iloc[0] - 2 * row["std"].iloc[0]
upper = row["mean"].iloc[0] + 2 * row["std"].iloc[0]

st.header("Current temperature analysis")
st.metric("Current temperature", f"{st.session_state.current_temp:.1f} °C")
st.write(f"Normal range: {lower:.1f} – {upper:.1f} °C")

if lower <= st.session_state.current_temp <= upper:
    st.success("Current temperature is within the seasonal norm")
else:
    st.error("Current temperature is anomalous for this season")

