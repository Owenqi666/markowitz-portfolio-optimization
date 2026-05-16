import numpy as np
from data import load_data, tickers
from optimize import (
    optimize_min_variance, optimize_max_sharpe, print_portfolio
)
from frontier import plot_frontier
from walkforward import run_walkforward
from shrinkage import ledoit_wolf_sigma, get_shrinkage_info

np.random.seed(42)

#step 1: load data and estimate parameters
prices, log_ret, miu, sigma, rf = load_data()

#step 2: full-sample optimization
print("\n" + "=" * 60)
print("FULL-SAMPLE OPTIMIZATION")
print("=" * 60)

for mode in ["unconstrained", "long_only", "long_only_capped"]:
    print(f"\n  --- {mode} ---")
    w_mvp = optimize_min_variance(miu, sigma, mode)
    print_portfolio(w_mvp, miu, sigma, rf, tickers, f"Min Variance ({mode})")

    w_tan = optimize_max_sharpe(miu, sigma, rf, mode)
    print_portfolio(w_tan, miu, sigma, rf, tickers, f"Max Sharpe ({mode})")

#step 3: efficient frontier plot
print("\n" + "=" * 60)
print("  EFFICIENT FRONTIER")
print("=" * 60)
plot_frontier(miu, sigma, rf, tickers)

#step 4: walk-forward with sample covariance
df_sample = run_walkforward(log_ret, rf, tickers, label="Sample")

#step 5: walk-forward with Ledoit-Wolf shrinkage
delta = get_shrinkage_info(log_ret.values)
print(f"\n  Ledoit-Wolf shrinkage intensity (full sample): {delta:.4f}")
df_shrink = run_walkforward(log_ret, rf, tickers,
                            sigma_estimator=ledoit_wolf_sigma,
                            label="Ledoit-Wolf")

#step 6: comparison
print("\n" + "=" * 60)
print("SAMPLE vs SHRINKAGE COMPARISON")
print("=" * 60)
print(f"{'Metric':<25} {'Sample':>10} {'Shrinkage':>10}")
print(f"{'─' * 45}")

metrics = [
    ("Mean OOS Sharpe", "opt_sharpe"),
    ("Mean OOS Return", "opt_ret"),
    ("Mean OOS Volatility", "opt_vol"),
    ("Mean OOS MDD", "opt_mdd"),
    ("Mean Turnover", "opt_turnover"),
]
for label, col in metrics:
    s_val = df_sample[col].mean()
    lw_val = df_shrink[col].mean()
    print(f"{label:<25} {s_val:>10.4f} {lw_val:>10.4f}")

#turnover reduction
s_turn = df_sample["opt_turnover"].mean()
lw_turn = df_shrink["opt_turnover"].mean()
if s_turn > 1e-6:
    pct_change = (lw_turn - s_turn) / s_turn * 100
    print(f"\n  Turnover change: {pct_change:+.1f}%")
