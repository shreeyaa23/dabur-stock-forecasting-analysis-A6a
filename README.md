# Dabur Stock Forecasting and Trend Analysis

This project analyzes historical stock data for Dabur India Ltd. (DABUR.BO) to identify trends, visualize patterns, and forecast future price movements.  
It applies time-series analysis techniques to evaluate performance, volatility, and investment potential using Python.

---

## Project Overview
The objective of this project is to perform a detailed statistical and visual analysis of Dabur India's stock performance using daily data obtained from Yahoo Finance.  
The analysis includes exploratory data analysis, trend evaluation, moving averages, and predictive modeling through time-series forecasting.

---

## Objectives
- Perform exploratory data analysis (EDA) on Dabur India Ltd. stock data.
- Identify stock trends and seasonal patterns using moving averages.
- Apply ARIMA-based time-series forecasting to predict future prices.
- Visualize historical and forecasted stock trends for better investment insights.

---

## Dataset
- **Source:** Yahoo Finance  
- **Ticker:** DABUR.BO  
- **Data Fields:** Date, Open, High, Low, Close, Adjusted Close, Volume  
- **Timeframe:** Multi-year daily stock data (historical closing prices)

---

## Methodology
1. **Data Cleaning and Preparation**
   - Handled missing values and formatted date columns.
   - Converted the dataset into a time-series structure.
2. **Exploratory Data Analysis (EDA)**
   - Analyzed price distributions and correlations.
   - Computed moving averages to smooth out volatility.
3. **Visualization**
   - Plotted trends, rolling means, and volatility bands.
   - Compared short-term vs. long-term stock trends.
4. **Forecasting**
   - Implemented ARIMA and rolling statistics for future price prediction.
   - Evaluated model performance through residual analysis and error metrics.

---

## Tools and Libraries
- **Python 3.10+**
- **Libraries:** pandas, numpy, matplotlib, seaborn, statsmodels

---

## Key Insights
- Identified a consistent long-term growth trend with periodic volatility.
- Observed seasonal fluctuations and cyclical price corrections.
- Forecast results indicate moderate upward movement for Dabur India Ltd. stock.
- Rolling mean and standard deviation plots suggest mean-reverting tendencies in short-term behavior.

---

## Results
The final analysis provided:
- Forecast plots showing predicted prices for the upcoming months.
- Statistical summary of moving averages and volatility bands.
- Time-series decomposition into trend, seasonal, and residual components.

---
