import numpy as np
import pandas as pd

TRADING_DAYS = 252


def rebalance_dates(index, freq="QE"):
    s = pd.Series(index=index, data=index)
    return set(s.resample(freq).last().dropna())


def backtest(prices, weight_fn, lookback=756, freq="QE", cost_bps=10.0):
    """
    Walk-forward backtest. weight_fn(window_returns) -> weights.

    No lookahead: on each date the portfolio first earns that day's return
    using weights decided earlier, and only then may rebalance using data
    up to and including that date.
    """
    rets = prices.pct_change().dropna(how="all")
    if len(rets) <= lookback:
        raise ValueError("not enough history for that lookback")

    dates = rets.index[lookback:]
    reb = rebalance_dates(rets.index, freq)
    reb.add(dates[0])

    w = pd.Series(0.0, index=rets.columns)
    out, weights, turnover = {}, {}, {}

    for d in dates:
        r = rets.loc[d]
        out[d] = float(w @ r)

        if w.sum() > 1e-9:                      # drift with the market
            w = w * (1 + r)
            w = w / w.sum()

        if d in reb:
            target = weight_fn(rets.loc[:d].iloc[-lookback:])
            t = float((target - w).abs().sum())
            out[d] -= t * cost_bps / 1e4
            turnover[d], weights[d] = t, target
            w = target

    return (pd.Series(out, name="ret"),
            pd.DataFrame(weights).T,
            pd.Series(turnover, name="turnover"))


def performance(rets: pd.Series, rf=0.02) -> dict:
    n = len(rets)
    cum = (1 + rets).cumprod()
    cagr = cum.iloc[-1] ** (TRADING_DAYS / n) - 1
    vol = rets.std() * np.sqrt(TRADING_DAYS)
    dd = cum / cum.cummax() - 1
    downside = rets[rets < 0].std() * np.sqrt(TRADING_DAYS)
    return {"cagr": round(float(cagr), 4),
            "vol": round(float(vol), 4),
            "sharpe": round(float((cagr - rf) / vol), 3),
            "sortino": round(float((cagr - rf) / downside), 3),
            "max_dd": round(float(dd.min()), 4),
            "calmar": round(float(cagr / abs(dd.min())), 3)}