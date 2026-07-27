import matplotlib.pyplot as plt
from quantfolio.config import TICKERS, UNIVERSE
from quantfolio.data.prices import fetch_prices, to_returns
from quantfolio.data.universe import align
from quantfolio.estimation import annualised_mean, sample_cov
from quantfolio.optimise import (Constraints, efficient_frontier,
                                 max_sharpe, summarise)

prices = align(fetch_prices(TICKERS), mode="common")
rets = to_returns(prices)
mu, cov = annualised_mean(rets), sample_cov(rets)

scenarios = {
    "unconstrained": Constraints(),
    "cap 25%": Constraints(w_max=0.25),
    "cap 25% + sector limits": Constraints(
        w_max=0.25,
        groups=UNIVERSE,
        group_limits={"equity_us": (0.10, 0.40),
                      "bonds": (0.20, 0.50),
                      "commodities": (0.00, 0.20),
                      "equity_intl": (0.05, 0.30)},
    ),
}

fig, ax = plt.subplots(figsize=(8, 5))
for name, cons in scenarios.items():
    ef = efficient_frontier(mu, cov, cons)
    ax.plot(ef["vol"], ef["ret"], lw=2, label=name)
    ms = max_sharpe(mu, cov, cons)
    print(f"\n--- {name} ---")
    print(ms[ms > 1e-4].round(3).sort_values(ascending=False))
    print(summarise(ms, mu, cov))

ax.set_xlabel("annualised volatility")
ax.set_ylabel("annualised return")
ax.legend()
plt.tight_layout()
plt.savefig("constrained.png", dpi=150)
print("\nsaved constrained.png")