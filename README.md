# Temperature Analysis & Monitoring App

An interactive Streamlit application for analyzing historical temperature data, detecting anomalies, and monitoring current temperature using the OpenWeatherMap API.

Live application:  
[https://temperature-analysis-app-app-i7vhng9febnt4venzkgwfu.streamlit.app/](https://temperature-analysis-app-app-i7vhng9febnt4venzkgwfu.streamlit.app/)

---

## Project Description

The application performs time series analysis of historical temperature data for multiple cities and compares current temperature values with historical seasonal norms. The project is implemented as an interactive Streamlit dashboard and deployed using Streamlit Cloud.

---

## Main Features

- Rolling mean and rolling standard deviation for temperature smoothing (configurable window: 30 / 60 / 90 days)
- Detection of temperature anomalies based on the rule: moving average ± 2σ
- Long-term temperature trend analysis using a 365-day moving average
- Seasonal statistics (mean and standard deviation)
- Seasonal temperature profile (bar chart with Mean ± Std)
- Number of anomaly days per season (table and bar chart)
- Current temperature monitoring via OpenWeatherMap API
- Comparison of synchronous and asynchronous API requests
- Multiprocessing experiment for historical data processing
- Export of detected anomalies to CSV
- Fully interactive visualizations

---

## Data Format

The application expects a CSV file with the following columns:

| Column | Description |
|------|-------------|
| city | City name |
| timestamp | Date (daily frequency) |
| temperature | Average daily temperature (°C) |
| season | Season (winter, spring, summer, autumn) |


