import numpy as np
from scipy.optimize import minimize

#portfolio metrics
def portfolio_return(w, miu):
    return w @ miu


def portfolio_volatility(w, sigma):
    return np.sqrt(w @ sigma @ w)


def neg_sharpe(w, miu, sigma, rf):
    ret = portfolio_return(w, miu)
    vol = portfolio_volatility(w, sigma)
    if vol < 1e-10:
        return 999
    return -(ret - rf) / vol


#constraint builders
def get_bounds(n, mode="unconstrained"):
    if mode == "unconstrained":
        return None
    elif mode == "long_only":
        return [(0, 1)] * n
    elif mode == "long_only_capped":
        return [(0, 0.25)] * n
    else:
        return None


#core optimization
def optimize_min_variance(miu, sigma, mode="unconstrained"):
    n = len(miu)
    w0 = np.ones(n) / n
    bounds = get_bounds(n, mode)
    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]

    result = minimize(
        lambda w: w @ sigma @ w,
        w0, method="SLSQP", bounds=bounds, constraints=cons
    )
    if not result.success:
        print(f"[Warning] MVP optimization failed: {result.message}")
    return result.x


def optimize_max_sharpe(miu, sigma, rf, mode="unconstrained"):
    n = len(miu)
    w0 = np.ones(n) / n
    bounds = get_bounds(n, mode)
    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]

    result = minimize(
        neg_sharpe, w0, args=(miu, sigma, rf),
        method="SLSQP", bounds=bounds, constraints=cons
    )
    if not result.success:
        print(f"[Warning] Max-Sharpe optimization failed: {result.message}")
    return result.x


def optimize_target_return(miu, sigma, target_ret, mode="unconstrained"):
    n = len(miu)
    w0 = np.ones(n) / n
    bounds = get_bounds(n, mode)
    cons = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1},
        {"type": "eq", "fun": lambda w: w @ miu - target_ret}
    ]

    result = minimize(
        lambda w: w @ sigma @ w,
        w0, method="SLSQP", bounds=bounds, constraints=cons
    )
    if not result.success:
        pass  # silent — frontier sweeps often hit infeasible edges
    return result.x


#print portfolio summary
def print_portfolio(w, miu, sigma, rf, tickers, label="Portfolio"):
    ret = portfolio_return(w, miu)
    vol = portfolio_volatility(w, sigma)
    sharpe = (ret - rf) / vol if vol > 1e-10 else 0

    print(f"\n {label}")
    print(f"{'─' * 40}")
    print(f"Return: {ret:.4f}  Volatility: {vol:.4f}  Sharpe: {sharpe:.4f}")
    for i, t in enumerate(tickers):
        if abs(w[i]) > 0.001:
            print(f"{t}: {w[i]:+.4f}")


if __name__ == "__main__":
    from data import load_data, tickers

    prices, log_ret, miu, sigma, rf = load_data()

    print("\n=== Unconstrained ===")
    w_mvp = optimize_min_variance(miu, sigma, "unconstrained")
    print_portfolio(w_mvp, miu, sigma, rf, tickers, "Min Variance (unconstrained)")

    w_tan = optimize_max_sharpe(miu, sigma, rf, "unconstrained")
    print_portfolio(w_tan, miu, sigma, rf, tickers, "Max Sharpe (unconstrained)")

    print("\n=== Long-Only ===")
    w_mvp_lo = optimize_min_variance(miu, sigma, "long_only")
    print_portfolio(w_mvp_lo, miu, sigma, rf, tickers, "Min Variance (long-only)")

    w_tan_lo = optimize_max_sharpe(miu, sigma, rf, "long_only")
    print_portfolio(w_tan_lo, miu, sigma, rf, tickers, "Max Sharpe (long-only)")

    print("\n=== Long-Only + Capped (25%) ===")
    w_tan_cap = optimize_max_sharpe(miu, sigma, rf, "long_only_capped")
    print_portfolio(w_tan_cap, miu, sigma, rf, tickers, "Max Sharpe (capped)")
