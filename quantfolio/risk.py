import numpy as np
import pandas as pd
from scipy.stats import norm, skew, kurtosis

TRADING_DAYS = 252

CRISES = {
    "GFC": ("2007-10-09", "2009-03-09"),
    "Euro crisis": ("2011-07-01", "2011-10-03"),
    "Taper tantrum": ("2013-05-22", "2013-09-05"),
    "COVID crash": ("2020-02-19", "2020-03-23"),
    "2022 rate shock": ("2022-01-03", "2022-10-12"),
}


# ---------- tail risk ----------

def historical_var(r, alpha=0.05):
    return -float(np.quantile(r, alpha))


def historical_cvar(r, alpha=0.05):
    q = np.quantile(r, alpha)
    return -float(r[r <= q].mean())


def gaussian_var(r, alpha=0.05):
    return -float(r.mean() + norm.ppf(alpha) * r.std())


def cornish_fisher_var(r, alpha=0.05):
    """Gaussian VaR adjusted for skew and excess kurtosis."""
    z = norm.ppf(alpha)
    s, k = skew(r), kurtosis(r)
    z_cf = (z + (z**2 - 1) * s / 6
              + (z**3 - 3*z) * k / 24
              - (2*z**3 - 5*z) * s**2 / 36)
    return -float(r.mean() + z_cf * r.std())


def tail_table(r, alpha=0.05) -> dict:
    return {"hist_VaR": round(historical_var(r, alpha), 4),
            "hist_CVaR": round(historical_cvar(r, alpha), 4),
            "gauss_VaR": round(gaussian_var(r, alpha), 4),
            "CF_VaR": round(cornish_fisher_var(r, alpha), 4),
            "skew": round(float(skew(r)), 3),
            "exc_kurt": round(float(kurtosis(r)), 3),
            "worst_day": round(float(r.min()), 4)}


# ---------- drawdown ----------

def drawdown_series(r: pd.Series) -> pd.Series:
    cum = (1 + r).cumprod()
    return cum / cum.cummax() - 1


def drawdown_episodes(r: pd.Series, top=5) -> pd.DataFrame:
    dd = drawdown_series(r)
    rows, start = [], None
    for d, v in dd.items():
        if v < 0 and start is None:
            start = d
        elif v >= 0 and start is not None:
            seg = dd.loc[start:d]
            rows.append({"start": start.date(), "trough": seg.idxmin().date(),
                         "recovered": d.date(), "depth": round(seg.min(), 4),
                         "days": len(seg)})
            start = None
    if start is not None:
        seg = dd.loc[start:]
        rows.append({"start": start.date(), "trough": seg.idxmin().date(),
                     "recovered": None, "depth": round(seg.min(), 4),
                     "days": len(seg)})
    return (pd.DataFrame(rows).sort_values("depth")
            .head(top).reset_index(drop=True))


# ---------- stress testing ----------

def stress_test(prices, weights, windows=CRISES) -> pd.DataFrame:
    """Apply fixed weights to historical crisis windows. Buy-and-hold within
    each window — no rebalancing, so this is a pure exposure test."""
    rets = prices.pct_change().dropna(how="all")
    w = weights.reindex(rets.columns).fillna(0.0)
    rows = {}
    for name, (s, e) in windows.items():
        seg = rets.loc[s:e]
        if len(seg) < 5:
            continue
        port = seg @ w
        rows[name] = {"days": len(port),
                      "total_return": round(float((1 + port).prod() - 1), 4),
                      "max_dd": round(float(drawdown_series(port).min()), 4),
                      "worst_day": round(float(port.min()), 4),
                      "ann_vol": round(float(port.std() * np.sqrt(TRADING_DAYS)), 4)}
    return pd.DataFrame(rows).T


# ---------- monte carlo ----------

def monte_carlo(rets, weights, horizon=TRADING_DAYS, n_paths=5000,
                method="block", block=21, seed=0):
    """Simulate terminal wealth. method: gaussian | iid | block."""
    rng = np.random.default_rng(seed)
    port = (rets @ weights.reindex(rets.columns).fillna(0.0)).values

    if method == "gaussian":
        sims = rng.normal(port.mean(), port.std(), (n_paths, horizon))
    elif method == "iid":
        sims = rng.choice(port, (n_paths, horizon), replace=True)
    elif method == "block":
        n_blocks = int(np.ceil(horizon / block))
        starts = rng.integers(0, len(port) - block, (n_paths, n_blocks))
        idx = starts[:, :, None] + np.arange(block)
        sims = port[idx].reshape(n_paths, -1)[:, :horizon]
    else:
        raise ValueError(f"unknown method: {method}")

    terminal = (1 + sims).prod(axis=1)
    return pd.Series(terminal, name=method)


def mc_summary(terminal: pd.Series) -> dict:
    return {"median": round(float(terminal.median()), 4),
            "p5": round(float(terminal.quantile(0.05)), 4),
            "p95": round(float(terminal.quantile(0.95)), 4),
            "P(loss)": round(float((terminal < 1).mean()), 4),
            "P(-20%)": round(float((terminal < 0.8).mean()), 4)}