import streamlit as st
import pandas as pd
import numpy as np
import pmdarima as pm
import os
import matplotlib.pyplot as plt

clean_path = os.path.join("..", "data", "processed")

# ---------- 1) FORECASTING ----------
def forecast_ticker(ticker, n_steps):
    file_path = os.path.join(clean_path, f"{ticker}")
    df = pd.read_csv(file_path, parse_dates = ["Date"], index_col = "Date", low_memory = False).asfreq("B").ffill()

    price_col = df["Adj Close"].dropna()
    returns_col = df["Return"].dropna()
    last_price = float(price_col.iloc[-1])

    model = pm.auto_arima(returns_col, start_p = 0, start_q = 0, seasonal = False, stepwise = True, suppress_warnings = True)

    mean_return, confidence_intervals = model.predict(n_periods = int(n_steps), return_conf_int = True, alpha = 0.05)
    index = pd.bdate_range(start = returns_col.index[-1] + pd.offsets.BDay(1), periods = int(n_steps), freq = "B")

    forecasted_returns = pd.DataFrame({"Predicted returns": mean_return, 
                                       "Lower bound of returns": confidence_intervals[:, 0],
                                       "Upper bound of returns": confidence_intervals[:, 1]},
                                       index=index)

    cumulative_return_product = (1 + forecasted_returns["Predicted returns"]).cumprod()
    forecasted_price = pd.DataFrame({"yhat": last_price * cumulative_return_product}, index = index)
    return forecasted_returns, forecasted_price

# ---------- 2) COVARIANCE MATRIX ----------

def get_returns_matrix(tickers, lookback_days = None):
    list = []
    for ticker in tickers:
        fp = os.path.join(clean_path, ticker if ticker.lower().endswith(".csv") else f"{ticker}.csv")
        df = pd.read_csv(fp, parse_dates = ["Date"], low_memory = False)
        df = df.set_index("Date").sort_index().asfreq("B").ffill()
        s = df["Return"].rename(ticker).dropna()
        if lookback_days is not None:
            s = s.iloc[-int(lookback_days):]
        list.append(s)
    return pd.concat(list, axis = 1).dropna(how = "any")

def get_cov_matrix(tickers, lookback_days = None):
    R = get_returns_matrix(tickers, lookback_days)
    return R.cov() * 252

def compute_portfolio_risk(weight_portfolio, cov_matrix):
    weights = np.asarray(weight_portfolio)
    cov = cov_matrix.values
    return np.sqrt(weights.T @ cov @ weights)

# ---------- 3) PORTFOLIO OPTIMISATION ----------
np.random.seed(42)

def get_weights(tickers, exp_ret = None, num_portfolios = 5000, lookback_days = None):
    returns_matrix = get_returns_matrix(tickers, lookback_days)
    names = list(returns_matrix.columns)
    mu_annual = pd.Series(exp_ret, index = names).astype(float)
    cov_annual = get_cov_matrix(tickers, lookback_days)

    n = len(names)
    rf = 0.01

    port_returns = []
    port_vol = []
    port_weights = []

    for _ in range(num_portfolios):
        weights = np.random.random(n)
        weights /= np.sum(weights)
        port_weights.append(weights)
        returns = float(weights @ mu_annual.values)
        port_returns.append(returns)
        port_vol.append(compute_portfolio_risk(weights, cov_annual))

    data = {"Returns": port_returns,
            "Risk": port_vol,
            "Sharpe Ratio": (np.array(port_returns) - rf) / np.array(port_vol)}
    
    for index, name in enumerate(names):
        data[name + " weight"] = [weight[index] for weight in port_weights]
    df = pd.DataFrame(data)
    return df

def get_optimal_weights(df):
    sr = df["Sharpe Ratio"].replace([np.inf, -np.inf], np.nan)
    best_idx = sr.idxmax()
    return df.loc[best_idx]

def row_to_weight_series(row):
    weight_cols = [c for c in row.index if c.endswith(" weight")]
    out = {c.replace(" weight", ""): float(row[c]) for c in weight_cols}
    s = pd.Series(out)
    return s / s.sum()

def strip_ext(name):
    return name[:-4] if name.lower().endswith(".csv") else name

# ---------- 4) BACKTESTING ----------

def metrics_from_returns(returns, rf_annual = 0.01):
    returns = returns.dropna()
    total_return = float((1 + returns).prod() - 1)
    risk = float(returns.std() * np.sqrt(252))
    sharpe = ((returns.mean() * 252) - rf_annual) / risk

    equity = (1 + returns).cumprod()
    peak = equity.cummax()
    drawdown = (equity / peak - 1.0)
    max_drawdown = float(drawdown.min())
    worst_day = float(returns.min())
    best_day = float(returns.max())

    return {"Total Return": total_return,
            "Volatility": risk,
            "Sharpe": sharpe,
            "Max Drawdown": max_drawdown,
            "Best Day": best_day,
            "Worst Day": worst_day}

def backtest(weights, start_date = None, end_date = None):
    tickers = list(weights.index)
    R = get_returns_matrix(tickers, lookback_days = None)
    R = R[R.index >= pd.to_datetime(start_date)]
    R = R[R.index <= pd.to_datetime(end_date)]

    R = R[weights.index]
    port_ret = (R * weights.values).sum(axis = 1)
    equity = (1 + port_ret).cumprod()
    equity_df = pd.DataFrame({"Portfolio performance": equity})

    metrics = metrics_from_returns(port_ret, rf_annual = 0.01)
    return equity_df, metrics

# ---------------- UI ----------------
st.set_page_config(page_title = "Predictive Portfolio Optimiser", layout="wide")
st.title("Predictive Portfolio Optimiser")


"""
Hello and welcome to the Predictive Portfolio Optimiser!
Here's how it works:
1. Choose 1 or more tickers from the S&P 100 list  
2. Choose an amount of days to see the forecast for  
3. Review the forecasted predictions for your choices on a graph  
5. Review your optimally weighted portfolio on a pie chart  
6. Review how the optimal portfolio would have performed historically for your chosen time period  
8. Review key metrics to see if the portfolio is sensible for the chosen period
"""

# Build available list and give a safe default selection
available = sorted([f for f in os.listdir(clean_path) if f.lower().endswith(".csv")])

col_left, col_right = st.columns([1, 2], gap = "large")

with col_left:
    st.subheader("Controls")
    tickers = st.multiselect("1) Select 1 or more tickers from the S&P 100 list", options = available)
    n_steps = st.number_input("2) Select an amount of days to forecast", min_value = 1, max_value = 20, value = 10, step = 1)
    st.caption("Tip: Think about how long you want to hold a single portfolio for!")
    lookback = st.number_input("Select a number of days to look back upon", min_value = 60, max_value = 252*5, value = 252, step = 14)
    st.caption("Tip: Think about how much historical data we should use to judge how risky these tickers are together...")

with col_right:
    if not tickers:
        st.info("Pick at least one ticker to continue.")
        st.stop()

    st.subheader("Forecasts")
    exp_daily = {}   # expected daily return from ARIMA (mean of n-day forecast)
    exp_annual = {}  # annualised expectation used in optimisation

    labels = [strip_ext(t) for t in tickers if isinstance(t, str) and strip_ext(t).strip()]
    forecast_table = st.tabs(labels)
    for t, tab in zip(tickers, forecast_table):
        with tab:
            fc_returns, fc_price = forecast_ticker(t, int(n_steps))
            mu_d = float(fc_returns["Predicted returns"].mean())
            exp_daily[t] = mu_d
            exp_annual[t] = mu_d * 252

            chart_df = pd.DataFrame({"Forecast Price": fc_price["yhat"].astype(float)})
            st.line_chart(chart_df)

            with st.expander("Show forecast returns"):
                st.dataframe(fc_returns.rename(columns={"Predicted returns": "mean",
                                                        "Lower bound of returns": "lower interval estimate",
                                                        "Upper bound of returns": "upper interval estimate"}))

    st.divider()

st.subheader("Optimise your Portfolio")
frontier = get_weights(tickers, exp_ret = exp_annual, lookback_days = int(lookback), num_portfolios = 5000)
best_row = get_optimal_weights(frontier)
weights_series = row_to_weight_series(best_row)

fig, ax = plt.subplots(figsize = (4, 4))
ax.pie(weights_series.values, labels = weights_series.index, autopct = "%1.1f%%", startangle = 90)
ax.axis("equal")
st.pyplot(fig)
st.write("Weights:", weights_series.to_frame("weight").style.format("{:.2%}"))

st.caption("We compute ARIMA forecasts for each ticker, convert to an annual expected return, sample many randomly generated portfolio combinations, and pick the one with the highest Sharpe ratio using the historical covariance. This has given you the optimal portfolio of the tickers that you chose above.")

st.divider()
st.subheader("Backtest your optimal portfolio")

full_R = get_returns_matrix(tickers)
if full_R.empty:
    st.warning("No overlapping data across the selected tickers. Try different tickers or reduce lookback.")
    st.stop()

min_d, max_d = full_R.index.min().date(), full_R.index.max().date()
bt_range = st.date_input("Choose your backtest time period:", value = (min_d, max_d), min_value = min_d, max_value = max_d)

if isinstance(bt_range, tuple) and len(bt_range) == 2:
    start_date, end_date = bt_range
else:
    start_date, end_date = min_d, max_d

if st.button("Run backtest"):
    equity_df, metrics = backtest(weights_series, start_date = start_date, end_date = end_date)
    st.line_chart(equity_df.rename(columns = {"Portfolio performance": "Equity Curve"}))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Return", f"{metrics['Total Return']:.1%}")
    c2.metric("Volatility (ann.)", f"{metrics['Volatility']:.1%}")
    c3.metric("Sharpe", f"{metrics['Sharpe']:.2f}")
    c4.metric("Max Drawdown", f"{metrics['Max Drawdown']:.1%}")

    d1, d2 = st.columns(2)
    d1.metric("Worst Day", f"{metrics['Worst Day']:.1%}")
    d2.metric("Best Day", f"{metrics['Best Day']:.1%}")
    st.caption("Use these metrics to judge if the portfolio is sensible for your chosen time period.")
