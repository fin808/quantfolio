import pandas as pd

from quantfolio.config import TICKERS
from quantfolio.data.prices import fetch_prices
from quantfolio.data.universe import align
from quantfolio.estimation import annualised_mean, sample_cov
from quantfolio.optimise import Constraints, min_variance, max_sharpe
from quantfolio.backtest import backtest
from quantfolio.factors import factor_panel, attribution

prices = align(fetch_prices(TICKERS), mode="common")
panel = factor_panel()
capped = Constraints(w_max=0.25)

strategies = {
    "1/N": lambda w: pd.Series(1 / w.shape[1], index=w.columns),
    "min vol": lambda w: min_variance(sample_cov(w), capped),
    "max sharpe": lambda w: max_sharpe(annualised_mean(w), sample_cov(w), capped),
}

for name, fn in strategies.items():
    r, _, _ = backtest(prices, fn)
    print(f"\n=== {name} ===")
    print(attribution(r, panel).to_string())

print("\n=== SPY (sanity check) ===")
spy = prices["SPY"].pct_change().dropna()
print(attribution(spy, panel).to_string())