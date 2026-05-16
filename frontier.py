import numpy as np
import matplotlib.pyplot as plt
from optimize import (
    optimize_min_variance, optimize_max_sharpe, optimize_target_return, 
    portfolio_return, portfolio_volatility
)

#efficient frontier computation
def compute_frontier(miu, sigma, mode="unconstrained", n_points=50):
    #anchor points
    w_mvp = optimize_min_variance(miu, sigma, mode)
    ret_min = portfolio_return(w_mvp, miu)
    ret_max = max(miu)

    target_rets = np.linspace(ret_min, ret_max, n_points)
    frontier_ret = []
    frontier_vol = []

    for tr in target_rets:
        w = optimize_target_return(miu, sigma, tr, mode)
        r = portfolio_return(w, miu)
        v = portfolio_volatility(w, sigma)
        #skip clearly failed optimizations
        if v < 1e-10 or v > 5:
            continue
        frontier_ret.append(r)
        frontier_vol.append(v)

    return np.array(frontier_vol), np.array(frontier_ret)


#plot
def plot_frontier(miu, sigma, rf, tickers, save_path="efficient_frontier.png"):
    fig, ax = plt.subplots(figsize=(10, 7))

    #unconstrained frontier
    vol_unc, ret_unc = compute_frontier(miu, sigma, "unconstrained")
    ax.plot(vol_unc, ret_unc, "b-", linewidth=2, label="Unconstrained")

    #long-only frontier
    vol_lo, ret_lo = compute_frontier(miu, sigma, "long_only")
    ax.plot(vol_lo, ret_lo, "r--", linewidth=2, label="Long-Only")

    #min variance portfolio (long-only)
    w_mvp = optimize_min_variance(miu, sigma, "long_only")
    mvp_ret = portfolio_return(w_mvp, miu)
    mvp_vol = portfolio_volatility(w_mvp, sigma)
    ax.scatter(mvp_vol, mvp_ret, marker="*", s=300, c="green", zorder=5, label="Min Variance")

    #max sharpe portfolio (long-only)
    w_tan = optimize_max_sharpe(miu, sigma, rf, "long_only")
    tan_ret = portfolio_return(w_tan, miu)
    tan_vol = portfolio_volatility(w_tan, sigma)
    ax.scatter(tan_vol, tan_ret, marker="D", s=150, c="gold", edgecolors="black", zorder=5, label="Max Sharpe")

    #capital market line
    cml_vol = np.linspace(0, max(vol_unc) * 1.1, 100)
    sharpe_tan = (tan_ret - rf) / tan_vol
    cml_ret = rf + sharpe_tan * cml_vol
    ax.plot(cml_vol, cml_ret, "k--", linewidth=1, alpha=0.6, label="CML")

    #individual assets
    for i, t in enumerate(tickers):
        vol_i = np.sqrt(sigma[i, i])
        ret_i = miu[i]
        ax.scatter(vol_i, ret_i, marker="o", s=60, c="gray", zorder=4)
        ax.annotate(t, (vol_i, ret_i), textcoords="offset points", xytext=(5, 5), fontsize=9)

    ax.set_xlabel("Annualized Volatility", fontsize=12)
    ax.set_ylabel("Annualized Return", fontsize=12)
    ax.set_title("Efficient Frontier", fontsize=14)
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"[Plot] saved to {save_path}")
    plt.close()


if __name__ == "__main__":
    from data import load_data, tickers

    prices, log_ret, miu, sigma, rf = load_data()
    plot_frontier(miu, sigma, rf, tickers)
