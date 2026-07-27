import matplotlib.pyplot as plt
import pandas as pd
from quantfolio.config import TICKERS, UNIVERSE
from quantfolio.data.prices import fetch_prices
from quantfolio.data.universe import align
from quantfolio.estimation import annualised_mean, sample_cov, shrunk_cov
from quantfolio.optimise import Constraints, min_variance, max_sharpe
from quantfolio.backtest import backtest, performance

prices = align(fetch_prices(TICKERS), mode="common")

capped = Constraints(w_max=0.25)


def equal_weight(window):
    return pd.Series(1.0 / window.shape[1], index=window.columns)


def min_vol(window):
    return min_variance(sample_cov(window), capped)


def max_sr(window):
    return max_sharpe(annualised_mean(window), sample_cov(window), capped)


def max_sr_shrunk(window):
    cov, _ = shrunk_cov(window)
    return max_sharpe(annualised_mean(window), cov, capped)


strategies = {"1/N": equal_weight, "min vol": min_vol,
              "max sharpe": max_sr, "max sharpe (LW)": max_sr_shrunk}

results, curves = {}, {}
for name, fn in strategies.items():
    r, w, t = backtest(prices, fn)
    results[name] = {**performance(r), "avg_turnover": round(t.mean(), 3)}
    curves[name] = (1 + r).cumprod()
    print(f"done: {name}")

print("\n" + pd.DataFrame(results).T.to_string())

fig, ax = plt.subplots(figsize=(9, 5))
for name, c in curves.items():
    ax.plot(c.index, c.values, lw=1.5, label=name)
ax.set_yscale("log")
ax.set_ylabel("growth of 1 (log scale)")
ax.legend()
plt.tight_layout()
plt.savefig("backtest.png", dpi=150)
print("saved backtest.png")