import numpy as np
from sklearn.covariance import LedoitWolf
from walkforward import run_walkforward, days

# ── Ledoit-Wolf covariance estimator ────────────────────────────────────
# LedoitWolf fits on daily returns, so we annualize after fitting.
# The shrinkage intensity delta is computed analytically by sklearn.

def ledoit_wolf_sigma(daily_ret):
    lw = LedoitWolf().fit(daily_ret)
    sigma_shrunk = lw.covariance_ * days
    return sigma_shrunk


def get_shrinkage_info(daily_ret):
    # return shrinkage intensity for reporting
    lw = LedoitWolf().fit(daily_ret)
    return lw.shrinkage_


if __name__ == "__main__":
    from data import load_data, tickers

    prices, log_ret, miu, sigma, rf, irx_series = load_data()

    # show shrinkage intensity on full sample
    delta = get_shrinkage_info(log_ret.values)
    print(f"\n  Ledoit-Wolf shrinkage intensity (full sample): {delta:.4f}")

    # run walk-forward with shrinkage
    df_shrink = run_walkforward(log_ret, rf, tickers,
                                irx_series=irx_series,
                                sigma_estimator=ledoit_wolf_sigma,
                                label="Ledoit-Wolf")
