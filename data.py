import sys
import numpy as np
import pandas as pd
import yfinance as yf

#config
tickers = ["SPY", "QQQ", "IWM", "TLT", "GLD", "VNQ", "EFA"]
start = "2015-01-01"
end = "2025-12-31"
days = 252


#risk-free rate
def get_risk_free_rate(start, end):

    try:
        irx = yf.download("^IRX", start=start, end=end, progress=False, auto_adjust=True)

        if irx.empty:
            sys.exit("Error: no ^IRX data returned.")

        irx_series = irx["Close"].squeeze().dropna() / 100
        rate = float(irx_series.mean())
        print(f"[Risk-free] mean 3M T-bill yield ({start} to {end}): {rate:.4f}")
        return rate, irx_series
    
    except Exception as e:
        sys.exit(f"Error: ^IRX download failed ({e}).")


#price data
def get_prices(tickers, start, end):
    #download one by one to avoid yfinance SQLite lock bug
    frames = {}

    for t in tickers:
        raw = yf.download(t, start=start, end=end, progress=False, auto_adjust=True)

        if raw.empty:
            sys.exit(f"Error: no data for {t}.")

        frames[t] = raw["Close"].squeeze()
        print(f"[Data] {t}: {len(raw)} rows")

    prices = pd.DataFrame(frames).dropna()

    if prices.empty:
        sys.exit("Error: no price data returned. Check tickers and internet connection.")

    print(f"[Data] {len(prices)} trading days, {len(tickers)} assets")
    print(f"[Data] date range: {prices.index[0].date()} to {prices.index[-1].date()}")
    return prices


#return estimation
def compute_returns(prices):
    #daily log returns
    log_ret = np.log(prices / prices.shift(1)).dropna()
    return log_ret


def estimate_params(log_ret):
    #annualized expected return vector
    miu = log_ret.mean().values * days

    #annualized covariance matrix
    sigma = log_ret.cov().values * days

    return miu, sigma


#entry point
def load_data():
    print("Loading data...")
    prices = get_prices(tickers, start, end)
    log_ret = compute_returns(prices)
    miu, sigma = estimate_params(log_ret)
    rf, irx_series = get_risk_free_rate(start, end)

    print(f"\n  Annualized returns:")
    for i, t in enumerate(tickers):
        print(f"    {t}: {miu[i]:.4f}")
    print(f"\n  Covariance matrix shape: {sigma.shape}")
    print(f"  Risk-free rate: {rf:.4f}")

    return prices, log_ret, miu, sigma, rf, irx_series


if __name__ == "__main__":
    prices, log_ret, miu, sigma, rf, irx_series = load_data()
