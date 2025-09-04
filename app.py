import riskfolio as rf
import streamlit as st
from pathlib import Path
import pandas as pd
import numpy as np
import pmdarima as pm
import os

clean_path = os.path.join("..", "data", "processed") 

# ---------- 1) FORECASTING ----------

def forecast_ticker(ticker, n_steps):
    path = os.path.join("..", "data", "processed") 
    file_path = os.path.join(path, f"{ticker}.csv")
    df = pd.read_csv(file_path, parse_dates=["Date"], index_col="Date", low_memory=False).asfreq("B").ffill()

    price_col = "Adj Close"

    returns = df["Return"].dropna()
    last_price = float(df[price_col].iloc[-1])

    model = pm.auto_arima(returns, start_p=0, start_q=0, max_p=3, max_q=3, d=None, seasonal=False, stepwise=True, suppress_warnings=True, n_jobs=1)

    vals, confi = model.predict(n_periods=int(n_steps), return_conf_int=True, alpha=0.05)
    idx = pd.bdate_range(start=returns.index[-1] + pd.offsets.BDay(1), periods=int(n_steps), freq="B")

    fc_returns = pd.DataFrame({"ret_hat": vals, "ret_lower": confi[:,0], "ret_upper": confi[:,1]}, index=idx)
    fc_returns.index.name = "Date"

    cumu = (1.0 + pd.Series(fc_returns["ret_hat"].values, index=idx)).cumprod()
    fc_price = pd.DataFrame({"yhat": last_price * cumu}, index=idx)
    fc_price.index.name = "Date"
    return fc_returns, fc_price

# ---------- 2) HISTORICAL RETURNS MATRIX (for Riskfolio) ----------

def get_cov_matrix(tickers, lookback_days=None):
    rets = []
    for ticker in tickers:
        df = pd.read_csv(clean_path, f"{ticker}.csv", parse_dates=["Date"]).dropna(subset=["Return"])
        df = df.set_index("Date").asfreq("B").ffill()
        if lookback_days is not None:
            df = df.iloc[-lookback_days:]
        rets.append(df["Return"].rename(ticker))
    MATRIX = pd.concat(rets, axis=1).dropna()
    return MATRIX
# ---------- 3) PORTFOLIO OPTIMISATION ----------
def optimise_portfolio(mu, Sigma, long_only=True):
    # Riskfolio expects returns matrix, but for custom mean/cov, use Portfolio object directly
    port = rf.Portfolio()
    port.assets_stats = {"mu": mu, "cov": Sigma}
    if long_only:
        port.bounds = (0, 1)
    else:
        port.bounds = (-1, 1)
    weights = port.optimisation(model="Classic", rm="MV", obj="Sharpe", rf=0, l=0)
    return weights

# ---------- 4) MONTE CARLO SIMULATION ----------
def monte_carlo_simulation(exp_daily_ret, daily_cov_mat, weights, n_steps, n_sims=2000, last_price=1.0):
    L = np.linalg.cholesky(daily_cov_mat.values)  
    
# reshape for matrix operations
    mu = exp_daily_ret.values.reshape(-1,1)
    w = weights.values.reshape(-1,1)

    final_vals = []
    for _ in range(n_sims):
        z = np.random.randn(len(mu_daily), n_steps)
        ret_mat = (mu + L @ z).T 
        port_ret = (ret_mat @ w).ravel()  
        path = last_price * np.cumprod(1.0 + port_ret)
        final_vals.append(path[-1])
    final_vals = np.array(final_vals)
    return final_vals

# ---------- 5) BACKTESTING ----------

def backtest(tickers):
    R = get_cov_matrix(tickers)  # ~2y to have a runway
    # monthly endpoints
    months = sorted(list(set((R.index.year, R.index.month) for _ in R.index)))
    # Build month start indices
    month_starts = R.resample("M").apply(lambda x: x.index[-1]).shift(1).dropna()  # last day prev month
    month_starts = month_starts.index

    equity = []
    dates = []
    val = 1.0
    for i in range(12):  # last 12 months
        if i+13 > len(month_starts): break
        # window: use previous 252B days for Sigma, previous 63B for mu (simple)
        end = month_starts[-(i+1)]
        start_cov = end - pd.tseries.offsets.BDay(252)
        start_mu  = end - pd.tseries.offsets.BDay(63)
        R_cov = R.loc[start_cov:end].dropna()
        R_mu  = R.loc[start_mu:end].dropna()
        if len(R_cov)<60 or len(R_mu)<20: continue

        mu_hat = R_mu.mean()           # daily mean vector
        Sigma  = R_cov.cov()           # daily covariance
        w = optimise_portfolio(mu_hat, Sigma, long_only=True)

        # apply weights to next month
        nxt_start = end + pd.tseries.offsets.BDay(1)
        nxt_end   = end + pd.tseries.offsets.MonthEnd(1)
        R_next = R.loc[nxt_start:nxt_end]
        if R_next.empty: continue
        port = (R_next @ w).fillna(0.0)
        val = val * float((1.0 + port).prod())
        equity.append(val); dates.append(R_next.index[-1])

    if not equity:
        return None
    df_eq = pd.DataFrame({"equity": equity}, index=pd.DatetimeIndex(dates))
    df_eq.index.name = "Date"
    return df_eq.sort_index()

# ---------------- UI ----------------

st.title("Predictive Portfolio Optimiser")

# choose tickers (multi), horizon, sims
all_files = sorted([ticker for ticker in os.listdir(clean_path) if ticker.endswith('.csv')])
default = ["AAPL"] if "AAPL" in all_files else all_files[:1]
tickers = st.multiselect("Tickers", options = all_files)
n = st.number_input("Days to forecast", min_value=1, max_value=60, value=10, step=1)
n_sims = st.slider("Monte Carlo simulations", min_value=500, max_value=5000, value=2000, step=500)

if st.button("Run"):
    if not tickers:
        st.warning("Pick at least one ticker.")
    else:
        # 1) Forecast each ticker (mean daily forecast = μ_i)
        mu_daily = {}
        last_prices = {}
        fc_price_dict = {}
        for t in tickers:
            fc_ret, fc_px = forecast_ticker(t, int(n))
            mu_daily[t] = float(fc_ret["ret_hat"].mean())     # expected daily return from forecast
            last_prices[t] = float(fc_px.iloc[0, 0] / (1.0 + fc_ret.iloc[0,0]))  # approx last price
            fc_price_dict[t] = fc_px

        # show the first ticker's price forecast
        first = tickers[0]
        st.subheader(f"Forecasted price: {first}")
        st.line_chart(fc_price_dict[first][["yhat"]])

        # 2) Build Σ from historical returns (stable)
        R = get_cov_matrix(tickers, lookback_days=252)
        Sigma = R.cov()  # daily covariance

        # 3) Optimise weights (max Sharpe, long-only clipped)
        mu_series = pd.Series(mu_daily).reindex(tickers)
        w = optimise_portfolio(mu_series, Sigma, long_only=True)
        st.subheader("Optimised weights (long-only, sum=1)")
        st.dataframe(w.round(4).to_frame())

        # 4) Monte Carlo on portfolio over next n days
        port_last_price = 1.0  # normalise
        finals = monte_carlo_simulation(mu_series, Sigma, w, int(n), int(n_sims), last_price=port_last_price)
        ret_dist = (finals / port_last_price) - 1.0
        var5 = np.percentile(ret_dist, 5)
        st.subheader("Monte Carlo (portfolio)")
        st.write(f"Expected n-day return (mean): {ret_dist.mean():.4f}")
        st.write(f"5% VaR (n-day): {var5:.4f}")
        st.bar_chart(pd.Series(ret_dist, name="n-day return").sort_values().reset_index(drop=True))

        # 5) Tiny backtest (12 months, monthly rebalance)
        bt = backtest(tickers)
        if bt is not None and len(bt) > 0:
            st.subheader("Backtest (12 months, monthly rebalance)")
            st.line_chart(bt)
        else:
            st.info("Backtest skipped (not enough data).")