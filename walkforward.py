import sys
import numpy as np
import pandas as pd
from optimize import optimize_max_sharpe, portfolio_return, portfolio_volatility

#config
days = 252
is_window = 3 * days #3-year in-sample
oos_window = 1 * days #1-year out-of-sample
rebal_freq = 63 #quarterly rebalance (~63 trading days)
fee_rate = 0.0005


#simulate OOS with quarterly rebalance
def simulate_oos(log_ret_oos, w_opt, rebal_freq, fee_rate):
    #convert log returns to simple returns for compounding
    simple_ret = np.exp(log_ret_oos) - 1
    n_days, n_assets = simple_ret.shape

    port_rets = []
    total_turnover = 0
    w_current = w_opt.copy()

    for d in range(n_days):
        #portfolio return for this day (weighted sum of asset returns)
        r_day = w_current @ simple_ret[d]
        port_rets.append(r_day)

        #drift weights after return realization
        w_drifted = w_current * (1 + simple_ret[d])
        w_drifted = w_drifted / w_drifted.sum()

        #quarterly rebalance
        if (d + 1) % rebal_freq == 0 and d < n_days - 1:
            turnover = np.sum(np.abs(w_opt - w_drifted))
            total_turnover += turnover
            # deduct fee from portfolio
            port_rets[-1] -= fee_rate * turnover
            w_current = w_opt.copy()
        else:
            w_current = w_drifted

    port_rets = np.array(port_rets)
    return port_rets, total_turnover


#compute metrics from daily return series
def compute_metrics(port_rets, rf):
    ann_ret = np.mean(port_rets) * days
    ann_vol = np.std(port_rets, ddof=1) * np.sqrt(days)
    sharpe = (ann_ret - rf) / ann_vol if ann_vol > 1e-10 else 0

    #max drawdown
    cum = np.cumprod(1 + port_rets)
    peak = np.maximum.accumulate(cum)
    dd = (cum - peak) / peak
    mdd = dd.min()

    return ann_ret, ann_vol, sharpe, mdd


#walk-forward engine
def run_walkforward(log_ret, rf, tickers, sigma_estimator=None, label="Sample"):
    n = len(log_ret)
    if n < is_window + oos_window:
        sys.exit(f"Error: need at least {is_window + oos_window} days, got {n}.")

    #build window boundaries
    windows = []
    t = is_window
    while t + oos_window <= n:
        windows.append((t - is_window, t, t + oos_window))
        t += oos_window

    print(f"\n{'=' * 60}")
    print(f"  Walk-Forward [{label}] — {len(windows)} windows")
    print(f"{'=' * 60}")

    records = []

    for i, (is_start, is_end, oos_end) in enumerate(windows):
        #in-sample estimation
        is_ret = log_ret.iloc[is_start:is_end].values
        miu_is = is_ret.mean(axis=0) * days

        if sigma_estimator is not None:
            sigma_is = sigma_estimator(is_ret)
        else:
            sigma_is = np.cov(is_ret, rowvar=False) * days

        #get OOS risk-free rate from full-period rf (simplification)
        #optimize: max-Sharpe, long-only
        w_opt = optimize_max_sharpe(miu_is, sigma_is, rf, mode="long_only")

        #OOS simulation
        oos_ret = log_ret.iloc[is_end:oos_end].values
        port_rets, turnover = simulate_oos(oos_ret, w_opt, rebal_freq, fee_rate)
        ann_ret, ann_vol, sharpe, mdd = compute_metrics(port_rets, rf)

        #equal-weight baseline
        ew_w = np.ones(len(tickers)) / len(tickers)
        ew_rets, ew_turn = simulate_oos(oos_ret, ew_w, rebal_freq, fee_rate)
        ew_ret, ew_vol, ew_sharpe, ew_mdd = compute_metrics(ew_rets, rf)

        #best single IS asset baseline
        best_asset_idx = np.argmax(miu_is)
        ba_w = np.zeros(len(tickers))
        ba_w[best_asset_idx] = 1.0
        ba_rets, _ = simulate_oos(oos_ret, ba_w, rebal_freq, fee_rate)
        ba_ret, ba_vol, ba_sharpe, ba_mdd = compute_metrics(ba_rets, rf)

        is_dates = log_ret.index[is_start].strftime("%Y-%m")
        oos_dates = log_ret.index[is_end].strftime("%Y-%m")
        oos_end_date = log_ret.index[min(oos_end - 1, len(log_ret) - 1)].strftime("%Y-%m")

        print(f"\n Window {i+1}: IS {is_dates}→{oos_dates} | OOS {oos_dates}→{oos_end_date}")
        print(f"Optimized — Ret: {ann_ret:+.4f} Vol: {ann_vol:.4f}  "
              f"Sharpe: {sharpe:+.4f}  MDD: {mdd:.4f}  Turnover: {turnover:.4f}")
        print(f"Equal-Wt   — Ret: {ew_ret:+.4f} Vol: {ew_vol:.4f}  "
              f"Sharpe: {ew_sharpe:+.4f}  MDD: {ew_mdd:.4f}")
        print(f"Best IS — Ret: {ba_ret:+.4f} Vol: {ba_vol:.4f}  "
              f"Sharpe: {ba_sharpe:+.4f} MDD: {ba_mdd:.4f}  "
              f"(asset: {tickers[best_asset_idx]})")

        #top 3 weights
        sorted_idx = np.argsort(w_opt)[::-1][:3]
        top3 = ", ".join(f"{tickers[j]}:{w_opt[j]:.2f}" for j in sorted_idx)
        print(f"Top weights: {top3}")

        records.append({
            "window": i + 1,
            "oos_start": oos_dates,
            "opt_ret": ann_ret, "opt_vol": ann_vol, "opt_sharpe": sharpe,
            "opt_mdd": mdd, "opt_turnover": turnover,
            "ew_ret": ew_ret, "ew_vol": ew_vol, "ew_sharpe": ew_sharpe,
            "ba_ret": ba_ret, "ba_sharpe": ba_sharpe,
            "ba_asset": tickers[best_asset_idx]
        })

    # summary
    df = pd.DataFrame(records)
    print(f"\n {'─' * 50}")
    print(f"Aggregate [{label}]:")
    print(f"Optimized — Mean Sharpe: {df['opt_sharpe'].mean():.4f}"
          f"Mean Ret: {df['opt_ret'].mean():.4f} Mean Vol: {df['opt_vol'].mean():.4f}")
    print(f"Equal-Wt — Mean Sharpe: {df['ew_sharpe'].mean():.4f}"
          f"Mean Ret: {df['ew_ret'].mean():.4f}  Mean Vol: {df['ew_vol'].mean():.4f}")
    print(f"Best IS — Mean Sharpe: {df['ba_sharpe'].mean():.4f}"
          f"Mean Ret: {df['ba_ret'].mean():.4f}")
    print(f"Mean Turnover: {df['opt_turnover'].mean():.4f}")

    return df


if __name__ == "__main__":
    from data import load_data, tickers

    prices, log_ret, miu, sigma, rf = load_data()
    run_walkforward(log_ret, rf, tickers)
