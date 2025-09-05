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
    df = pd.read_csv(file_path, parse_dates=["Date"], index_col="Date", low_memory=False).asfreq("B").ffill()

    price_col = "Adj Close"
    returns = df["Return"].dropna()
    last_price = float(df[price_col].iloc[-1])

    model = pm.auto_arima(returns, start_p = 0, start_q = 0, max_p = 3, max_q = 3, seasonal = False, stepwise = True, suppress_warnings = True, n_jobs = 1)

    vals, confi = model.predict(n_periods = int(n_steps), return_conf_int = True, alpha = 0.05)
    index = pd.bdate_range(start = returns.index[-1] + pd.offsets.BDay(1), periods = int(n_steps), freq = "B")

    fc_returns = pd.DataFrame({"meanreturn": vals, "ret_lower": confi[:,0], "ret_upper": confi[:,1]}, index=index)

    cumu = (1.0 + pd.Series(fc_returns["meanreturn"].values, index=index)).cumprod()
    fc_price = pd.DataFrame({"yhat": last_price * cumu}, index=index)
    return fc_returns, fc_price

# ---------- 2) COVARIANCE MATRIX ----------

def get_returns_matrix(tickers, lookback_days=None):
    returns = []
    for ticker in tickers:
        df = pd.read_csv(os.path.join(clean_path, f"{ticker}"))
        df = df.iloc[-lookback_days:]
        returns.append(df["Return"].rename(ticker))
    matrix = (pd.concat(returns, axis=1).dropna())
    return matrix

def get_cov_matrix(tickers, lookback_days=None):
    R = get_returns_matrix(tickers, lookback_days)
    return R.cov() * 252

def compute_portfolio_risk(weight_portfolio, cov_matrix):
    weights = np.asarray(weight_portfolio)
    cov = cov_matrix.values
    return np.sqrt(weights.T @ cov @ weights)

# ---------- 3) PORTFOLIO OPTIMISATION ----------
np.random.seed(42)

def get_weights(tickers, exp_ret = None, num_portfolios = 5000, lookback_days=None):
    R = get_returns_matrix(tickers, lookback_days)
    names = list(R.columns)
    mu_annual = pd.Series(exp_ret, index=names).astype(float)
    cov_annual = get_cov_matrix(tickers, lookback_days) 
    
    n = len(names)
    rf = 0.01

    port_returns = []
    port_volatility = []
    port_weights = []
    
    for _ in range(num_portfolios):
        weights = np.random.random(n)
        weights /= np.sum(weights)
        port_weights.append(weights)
        returns = float(weights @ mu_annual.values)
        port_returns.append(returns)
        port_volatility.append(compute_portfolio_risk(weights, cov_annual))
        
    data = {"Returns": port_returns, "Volatility": port_volatility, "Sharpe Ratio": (np.array(port_returns) - rf) / np.array(port_volatility)}
    for index, symbol in enumerate(names):
        data[symbol + " weight"] = [w[index] for w in port_weights]
    df = pd.DataFrame(data)
    return df

def get_optimal_weights(df):
    rf = 0.01
    sharpe = df.iloc[((df["Returns"] - rf) / df["Volatility"])]
    return df.iloc[sharpe.idxmax()]

def row_to_weight_series(row: pd.Series) -> pd.Series:
    """Extract {ticker: weight} from a row returned by get_optimal_weights."""
    weight_cols = [c for c in row.index if c.endswith(" weight")]
    out = {c.replace(" weight", ""): float(row[c]) for c in weight_cols}
    s = pd.Series(out)
    # Normalise for safety
    return s / s.sum()

def strip_ext(name: str) -> str:
    return name[:-4] if name.lower().endswith(".csv") else name

# ---------- 4) BACKTESTING ----------

def metrics_from_returns(ret: pd.Series, rf_annual=0.01) -> dict:
    ret = ret.dropna()
    if ret.empty:
        return {}
    N = ret.shape[0]
    total_return = float((1 + ret).prod() - 1)
    years = N / 252
    cagr = (1 + total_return) ** (1 / years) - 1 
    vol = float(ret.std() * np.sqrt(252))
    sharpe = ((ret.mean() * 252) - rf_annual) / vol 

    # Max drawdown from equity curve
    equity = (1 + ret).cumprod()
    peak = equity.cummax()
    dd = (equity / peak - 1.0)
    max_dd = float(dd.min())
    worst_day = float(ret.min())
    best_day = float(ret.max())

    return {
        "Total Return": total_return,
        "CAGR": cagr,
        "Volatility": vol,
        "Sharpe": sharpe,
        "Max Drawdown": max_dd,
        "Best Day": best_day,
        "Worst Day": worst_day,
    }

def backtest(weights: pd.Series, start_date=None, end_date=None) -> tuple[pd.DataFrame, dict]:
    """
    weights: Series like {'AAPL': 0.3, 'MSFT': 0.7}
    Returns (equity_df, metrics_dict).
    """
    tickers = list(weights.index)
    R = get_returns_matrix(tickers, lookback_days=None)

    if start_date:
        R = R[R.index >= pd.to_datetime(start_date)]
    if end_date:
        R = R[R.index <= pd.to_datetime(end_date)]

    # align order
    R = R[weights.index]
    port_ret = (R * weights.values).sum(axis=1)
    equity = (1 + port_ret).cumprod()
    equity_df = pd.DataFrame({"Portfolio": equity})

    metrics = metrics_from_returns(port_ret, rf_annual=0.01)
    return equity_df, metrics


# ---------------- UI ----------------
st.set_page_config(page_title="Predictive Portfolio Optimiser", layout="wide")
st.title("Predictive Portfolio Optimiser")

'''
Hello and welcome to the Predictive Portfolio Optimiser!
Here's how it works:
1. Chooses 1 or more tickers from the S&P 100 list  
2. Choose an amount of days to see the forecast for
3. Review the forecasted predictions for your choices on the a graph 
5. Review your optimally weighted portfolio on the a pie chart
6. Review how the optimal portfolio would have performed historically for your chosen time period
8. Review key metrics to see if the portfolio is sensible for the chosen period
'''

# 1) User chooses tickers
for ticker in os.listdir(clean_path):
    available = [f for f in os.listdir(clean_path) if f.lower().endswith(".csv")]
    break

col_left, col_right = st.columns([1, 2], gap="large")

with col_left:
    st.subheader("Controls")
    tickers = st.multiselect("1) Select 1 or more tickers from the S&P 100 list", options=available)
    n_steps = st.number_input("2) Select an amount of days to forecast", min_value=1, max_value=60, value=10, step=1)
    st.caption("Tip: Think about how long you want to hold a single portfolio for!")
    lookback = st.number_input("Select a covariance lookback (number of days)", min_value=60, max_value=252*5, value=252, step=14)
    st.caption("Tip: Think about how much historical data we should use to judge how risky these tickers are together...")

with col_right:
    if not tickers:
        st.info("Pick at least one ticker to continue.")
        st.stop()

    st.subheader("Forecasts")
    exp_daily = {}   # expected daily return from ARIMA (mean of n-day forecast)
    exp_annual = {}  # annualised expectation used in optimisation

    fc_tab = st.tabs([strip_ext(t) for t in tickers])
    for t, tab in zip(tickers, fc_tab):
        with tab:
            fc_returns, fc_price = forecast_ticker(t, int(n_steps))
            # store expectations
            mu_d = float(fc_returns["meanreturn"].mean())
            exp_daily[strip_ext(t)] = mu_d
            exp_annual[strip_ext(t)] = mu_d * 252  # simple annualisation

            chart_df = pd.DataFrame({
                "Forecast Price": fc_price["yhat"].astype(float)
            })
            st.line_chart(chart_df)

            with st.expander("Show forecast returns (mean ± CI)"):
                st.dataframe(fc_returns.rename(columns={
                    "meanreturn": "mean",
                    "ret_lower": "lower",
                    "ret_upper": "upper"
                }))

    st.divider()
    st.divider()
    st.divider()
    st.subheader("Optimise your Portfolio")
    frontier = get_weights(tickers, exp_ret=exp_annual, lookback_days=int(lookback), num_portfolios=5000)
    best_row = get_optimal_weights(frontier)
    weights_series = row_to_weight_series(best_row)

    # 5) Display weights with a pie chart
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.pie(weights_series.values, labels=weights_series.index, autopct="%1.1f%%", startangle=90)
    ax.axis("equal")
    st.pyplot(fig)
    st.write("Weights:", weights_series.to_frame("weight").style.format("{:.2%}"))

    st.caption("We compute ARIMA forecasts for each ticker, convert to an annual expected return, "
               "sample many long-only weight vectors, and pick the one with the highest Sharpe ratio using the "
               f"historical covariance (lookback={int(lookback)} days).")

    # 6–8) Backtest option
    st.divider()
    st.subheader("6–8) Backtest the optimal portfolio")

    full_R = get_returns_matrix(tickers)
    min_d, max_d = full_R.index.min().date(), full_R.index.max().date()
    bt_range = st.date_input(
        "Choose backtest window (must lie within the intersection of ticker histories)",
        value=(min_d, max_d),
        min_value=min_d,
        max_value=max_d
    )

    if isinstance(bt_range, tuple) and len(bt_range) == 2:
        start_date, end_date = bt_range
    else:
        start_date, end_date = min_d, max_d

    if st.button("Run backtest"):
        equity_df, metrics = backtest(weights_series, start_date=start_date, end_date=end_date)
        st.line_chart(equity_df.rename(columns={"Portfolio": "Equity Curve"}))

        mcol1, mcol2, mcol3, mcol4 = st.columns(4)
        mcol1.metric("Total Return", f"{metrics['Total Return']:.1%}")
        mcol2.metric("CAGR", f"{metrics['CAGR']:.1%}")
        mcol3.metric("Volatility (ann.)", f"{metrics['Volatility']:.1%}")
        mcol4.metric("Sharpe", f"{metrics['Sharpe']:.2f}")

        mcol5, mcol6 = st.columns(2)
        mcol5.metric("Max Drawdown", f"{metrics['Max Drawdown']:.1%}")
        mcol6.metric("Worst Day", f"{metrics['Worst Day']:.1%}")
        st.caption("Use these metrics to judge if the portfolio is sensible for the chosen period.")