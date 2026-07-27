import matplotlib.pyplot as plt
from quantfolio.config import TICKERS
from quantfolio.data.prices import fetch_prices, to_returns
from quantfolio.data.universe import align
from quantfolio.estimation import annualised_mean, sample_cov
from quantfolio.optimise import (efficient_frontier, min_variance,
                                 max_sharpe, summarise)

prices = align(fetch_prices(TICKERS), mode="common")
rets = to_returns(prices)
mu, cov = annualised_mean(rets), sample_cov(rets)

ef = efficient_frontier(mu, cov)
mv = min_variance(cov)
ms = max_sharpe(mu, cov)

mv_stats = summarise(mv, mu, cov)
ms_stats = summarise(ms, mu, cov)

print("\n--- min variance ---")
print(mv[mv > 1e-4].round(4).sort_values(ascending=False))
print(mv_stats)

print("\n--- max sharpe ---")
print(ms[ms > 1e-4].round(4).sort_values(ascending=False))
print(ms_stats)

asset_vol = rets.std() * (252 ** 0.5)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(ef["vol"], ef["ret"], lw=2, label="efficient frontier")
ax.scatter(asset_vol, mu, s=25, c="grey", label="assets")
for t in mu.index:
    ax.annotate(t, (asset_vol[t], mu[t]), fontsize=8)
ax.scatter(mv_stats["vol"], mv_stats["ret"], marker="*", s=200,
           c="tab:blue", label="min vol", zorder=5)
ax.scatter(ms_stats["vol"], ms_stats["ret"], marker="*", s=200,
           c="tab:red", label="max sharpe", zorder=5)
ax.set_xlabel("annualised volatility")
ax.set_ylabel("annualised return")
ax.legend()
plt.tight_layout()
plt.savefig("frontier.png", dpi=150)
print("\nsaved frontier.png")