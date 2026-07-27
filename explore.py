from quantfolio.config import TICKERS
from quantfolio.data.prices import fetch_prices, to_returns
from quantfolio.data.universe import interior_gaps, align

prices = fetch_prices(TICKERS)
rets = to_returns(prices)

print("\n--- shape ---")
print(prices.shape)

print("\n--- first valid date per ticker ---")
print(prices.apply(lambda c: c.first_valid_index()).sort_values())

print("\n--- missing values ---")
print(prices.isna().sum())

print("\n--- annualised vol ---")
print((rets.std() * (252 ** 0.5)).sort_values().round(3))

print("\n--- interior gaps ---")
print(interior_gaps(prices))

aligned = align(prices, mode="common")
print("\n--- aligned ---")
print(aligned.shape, aligned.index[0].date(), "to", aligned.index[-1].date())

from quantfolio.estimation import (annualised_mean, sample_cov, ewma_cov,
                                   shrunk_cov, condition_number)

ar = to_returns(aligned)

print("\n--- annualised return ---")
print(annualised_mean(ar).sort_values().round(4))

S = sample_cov(ar)
E = ewma_cov(ar)
L, intensity = shrunk_cov(ar)

print("\n--- condition numbers ---")
for name, c in [("sample", S), ("ewma", E), ("ledoit-wolf", L)]:
    print(f"{name:12s} {condition_number(c):10.1f}")
print(f"shrinkage intensity: {intensity:.4f}")

print("\n--- correlation, selected ---")
corr = ar.corr()
print(corr.loc[["SPY", "TLT", "HYG", "GLD"], ["SPY", "TLT", "HYG", "GLD"]].round(2))