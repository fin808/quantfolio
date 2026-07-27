from dataclasses import dataclass, field

import cvxpy as cp
import numpy as np
import pandas as pd


@dataclass
class Constraints:
    """Portfolio constraints. Groups map a label to a list of tickers."""
    w_max: float = 1.0
    w_min: float = 0.0
    groups: dict = field(default_factory=dict)
    group_limits: dict = field(default_factory=dict)   # label -> (lo, hi)

    def build(self, w, assets, scale=1.0):
        """
        Constraint list for cvxpy. `scale` is 1.0 for a normal problem, or the
        auxiliary variable k for the max-Sharpe reformulation — every
        constraint must be homogeneous of degree 1 for that to work.
        """
        cons = [cp.sum(w) == scale,
                w >= self.w_min * scale,
                w <= self.w_max * scale]
        pos = {a: i for i, a in enumerate(assets)}
        for label, (lo, hi) in self.group_limits.items():
            members = [pos[t] for t in self.groups.get(label, []) if t in pos]
            if not members:
                continue
            held = cp.sum(w[members])
            cons += [held >= lo * scale, held <= hi * scale]
        return cons


def _solve(prob):
    prob.solve(solver=cp.CLARABEL)
    if prob.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"solver status: {prob.status}")


def min_variance(cov, cons=None) -> pd.Series:
    cons = cons or Constraints()
    w = cp.Variable(len(cov))
    prob = cp.Problem(cp.Minimize(cp.quad_form(w, cp.psd_wrap(cov.values))),
                      cons.build(w, list(cov.index)))
    _solve(prob)
    return pd.Series(w.value, index=cov.index)


def max_return(mu, cons=None) -> float:
    cons = cons or Constraints()
    w = cp.Variable(len(mu))
    prob = cp.Problem(cp.Maximize(mu.values @ w), cons.build(w, list(mu.index)))
    _solve(prob)
    return float(mu.values @ w.value)


def efficient_frontier(mu, cov, cons=None, n_points=40) -> pd.DataFrame:
    cons = cons or Constraints()
    assets = list(mu.index)
    lo = float(mu @ min_variance(cov, cons))
    hi = max_return(mu, cons)
    rows = []
    for target in np.linspace(lo, hi, n_points):
        w = cp.Variable(len(mu))
        prob = cp.Problem(
            cp.Minimize(cp.quad_form(w, cp.psd_wrap(cov.values))),
            cons.build(w, assets) + [mu.values @ w >= target],
        )
        try:
            _solve(prob)
        except RuntimeError:
            continue
        wt = pd.Series(w.value, index=mu.index)
        rows.append({"ret": float(mu @ wt),
                     "vol": float(np.sqrt(wt @ cov @ wt)),
                     **wt.to_dict()})
    return pd.DataFrame(rows)


def max_sharpe(mu, cov, cons=None, rf=0.02) -> pd.Series:
    cons = cons or Constraints()
    n = len(mu)
    y, k = cp.Variable(n), cp.Variable(nonneg=True)
    excess = mu.values - rf
    prob = cp.Problem(
        cp.Minimize(cp.quad_form(y, cp.psd_wrap(cov.values))),
        cons.build(y, list(mu.index), scale=k) + [excess @ y == 1],
    )
    _solve(prob)
    return pd.Series(y.value / k.value, index=mu.index)


def summarise(w, mu, cov, rf=0.02) -> dict:
    r = float(mu @ w)
    v = float(np.sqrt(w @ cov @ w))
    held = w[w > 1e-4]
    return {"ret": round(r, 4), "vol": round(v, 4),
            "sharpe": round((r - rf) / v, 3),
            "n_holdings": len(held),
            "effective_n": round(1 / (w ** 2).sum(), 1),
            "max_weight": round(w.max(), 3)}