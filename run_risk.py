import matplotlib.pyplot as plt
import pandas as pd

from quantfolio.config import TICKERS
from quantfolio.data.prices import fetch_prices, to_returns
from quantfolio.data.universe import align
from quantfolio.estimation import annualised_mean, sample_cov
from quantfolio.optimise import Constraints, min_variance, max_sharpe
from quantfolio.backtest import backtest, performance
from quantfolio.risk import (tail_table, drawdown_episodes, stress_test,
                             monte_carlo, mc_summary)

prices = align(fetch_prices(TICKERS), mode="common")
rets = to_returns(prices)
mu, cov = annualised_mean(rets), sample_cov(rets)
capped = Constraints(w_max=0.25)

portfolios = {
    "1/N": pd.Series(1 / len(TICKERS), index=prices.columns),
    "min vol": min_variance(cov, capped),
    "max sharpe": max_sharpe(mu, cov, capped),
}

print("\n=== stress tests (fixed weights, buy-and-hold) ===")
for name, w in portfolios.items():
    print(f"\n{name}")
    print(stress_test(prices, w).to_string())

print("\n=== realised backtest risk ===")
tails, curves = {}, {}
for name, w in portfolios.items():
    fn = (lambda win, w=w: w) if name == "1/N" else None
    if name == "1/N":
        r, _, _ = backtest(prices, lambda win: pd.Series(1/win.shape[1], index=win.columns))
    elif name == "min vol":
        r, _, t = backtest(prices, lambda win: min_variance(sample_cov(win), capped))
        print(f"{name} avg turnover: {t.mean():.3f}")
    else:
        r, _, t = backtest(prices, lambda win: max_sharpe(annualised_mean(win),
                                                          sample_cov(win), capped))
        print(f"{name} avg turnover: {t.mean():.3f}")
    tails[name] = {**tail_table(r), **performance(r)}
    curves[name] = r

print("\n" + pd.DataFrame(tails).T.to_string())

print("\n=== worst drawdowns: max sharpe ===")
print(drawdown_episodes(curves["max sharpe"]).to_string())

print("\n=== monte carlo, 1y horizon, max sharpe ===")
mc = {}
for method in ["gaussian", "iid", "block"]:
    term = monte_carlo(rets, portfolios["max sharpe"], method=method)
    mc[method] = mc_summary(term)
    if method == "block":
        block_term = term
print(pd.DataFrame(mc).T.to_string())

fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(block_term, bins=80, color="tab:blue", alpha=0.8)
ax.axvline(1.0, color="k", lw=1)
ax.axvline(block_term.quantile(0.05), color="tab:red", lw=1.5, ls="--",
           label="5th percentile")
ax.set_xlabel("terminal wealth after 1 year")
ax.legend()
plt.tight_layout()
plt.savefig("montecarlo.png", dpi=150)
print("\nsaved montecarlo.png")