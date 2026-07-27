# Quantfolio

Multi-asset portfolio construction and risk analytics engine. Convex
optimisation under real-world constraints, walk-forward backtesting,
tail-risk analytics, and Fama-French factor attribution.

**Live app:** [add your Streamlit URL]

## Universe

13 ETFs spanning US and international equity, Treasuries, investment-grade
and high-yield credit, commodities, gold and REITs. Daily total-return
prices from Yahoo Finance, 2007-04-11 to present (4,853 observations).
Start date is set by the latest asset inception (HYG); the coverage
trade-off is parameterised rather than hidden in a `dropna()`.

## Findings

**1. Diversification is measurable.** The minimum-variance portfolio
realises 5.3% annualised volatility — below the 6.7% of IEF, the least
volatile individual asset. The gap is pure correlation benefit.

**2. Constraints are nearly free where it matters.** Capping positions at
25% costs 6.5% of in-sample Sharpe (0.78 → 0.73) but raises effective
holdings from 2.7 to 4.7. The cost is concentrated in the high-return
region of the frontier, which is precisely where the optimiser leans
hardest on unreliable mean estimates. Sector limits are a worse deal:
a further 5.5% of Sharpe for 0.5 effective holdings, almost all of it
from a mandate floor forcing exposure to an asset the optimiser rejects.

**3. The 2022 rate shock hurt more than the GFC.** The max-Sharpe
portfolio (IEF / QQQ / GLD) lost 20.5% through 2022 against 9.2% in
2007-09, and took 423 days to recover. The portfolio's diversification
depends on a negative stock-bond correlation estimated over a
low-inflation regime. When inflation returned, both legs fell together.
Minimum-variance lost only 5.9% through the GFC — the flight-to-quality
Treasury rally — but 15.4% in 2022, for the same reason.

**4. No alpha.** Regressing walk-forward strategy returns on Fama-French
factors with Newey-West standard errors: equal-weight shows alpha
indistinguishable from zero (t = -0.32 to 0.08). Max-Sharpe shows 3.1%
annualised alpha at t ≈ 1.8, which does not clear the ~3.0 threshold
Harvey, Liu and Zhu (2016) propose for multiple-testing-adjusted
significance — and with R² of only 0.44, most of the intercept is
unspanned bond and commodity premia rather than skill. The equity
factors cannot span this universe. Returns are factor exposure, not
selection.

## Walk-forward results

3-year rolling estimation, quarterly rebalancing, 10bps transaction
costs, 25% position cap. Out-of-sample 2010-2026.

| | CAGR | Vol | Sharpe | Max DD | Turnover/qtr |
|---|---|---|---|---|---|
| 1/N | 7.5% | 10.8% | 0.51 | -23.8% | — |
| Min variance | 5.6% | 6.1% | 0.59 | -17.1% | 8.7% |
| Max Sharpe | 9.0% | 9.5% | 0.74 | -20.9% | 34.1% |

Max-Sharpe's advantage survives realistic costs: 34% quarterly turnover
implies ~14bps/year at 10bps, against a 149bps CAGR edge over 1/N.
Caveat: the out-of-sample window is unusually favourable to a
growth-tilted portfolio.

## Methods

- **Optimisation** — cvxpy/Clarabel. Max-Sharpe via the standard convex
  reformulation (minimise `y'Σy` s.t. `(μ-r)'y = 1`, `Σy = κ`), with
  constraints scaled by κ to preserve homogeneity.
- **Covariance** — sample, EWMA, and Ledoit-Wolf shrinkage. Shrinkage
  intensity is ~0.005 at 13 assets; it matters at higher dimension.
- **Backtest** — returns are earned before rebalancing within each loop
  iteration, so no lookahead. Weights drift with the market between
  rebalances.
- **Risk** — historical, Gaussian and Cornish-Fisher VaR; CVaR;
  drawdown episodes; fixed-weight stress tests against five historical
  crisis windows; Monte Carlo via Gaussian, iid bootstrap, and block
  bootstrap (21-day blocks, preserving volatility clustering).
- **Attribution** — CAPM, FF3, Carhart 4, FF5 (+MOM), OLS with
  Newey-West HAC standard errors. SPY is used as a control: beta 0.974,
  R² 0.983, alpha ≈ 0.

## Install

```bash
git clone git@github.com:fin808/quantfolio.git
cd quantfolio
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
```

## Run

```bash
streamlit run app.py      # interactive frontier, weights, backtest
python explore.py         # data quality checks
python compare.py         # cost of constraints
python run_backtest.py    # walk-forward comparison
python run_risk.py        # VaR, stress tests, Monte Carlo
python run_factors.py     # factor attribution
```

Price and factor data are cached to `data/cache/` as parquet on first
run.

## Limitations

- ETF universe avoids survivorship bias but is not an equity-selection
  problem; extending to S&P 500 constituents would require delisting
  and ticker-change handling.
- Transaction costs are a flat 10bps with no market-impact model.
- Expected returns are historical means. Their standard errors are large
  enough that the max-Sharpe portfolio should be read as an illustration
  of estimation error, not a recommendation.