import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

TRADING_DAYS = 252


def annualised_mean(rets: pd.DataFrame, method: str = "geometric") -> pd.Series:
    """Annualised expected return per asset."""
    if method == "arithmetic":
        return rets.mean() * TRADING_DAYS
    if method == "geometric":
        return (1 + rets).prod() ** (TRADING_DAYS / len(rets)) - 1
    raise ValueError(f"unknown method: {method}")


def sample_cov(rets: pd.DataFrame) -> pd.DataFrame:
    """Plain sample covariance, annualised."""
    return rets.cov() * TRADING_DAYS


def ewma_cov(rets: pd.DataFrame, halflife: int = 126) -> pd.DataFrame:
    """Exponentially weighted covariance — recent data counts more."""
    r = rets - rets.mean()
    w = 0.5 ** (np.arange(len(r))[::-1] / halflife)
    w /= w.sum()
    X = r.values * np.sqrt(w)[:, None]
    return pd.DataFrame(X.T @ X * TRADING_DAYS,
                        index=rets.columns, columns=rets.columns)


def shrunk_cov(rets: pd.DataFrame):
    """Ledoit-Wolf shrinkage. Returns (covariance, shrinkage_intensity)."""
    lw = LedoitWolf().fit(rets.values)
    cov = pd.DataFrame(lw.covariance_ * TRADING_DAYS,
                       index=rets.columns, columns=rets.columns)
    return cov, lw.shrinkage_


def condition_number(cov: pd.DataFrame) -> float:
    """Ratio of largest to smallest eigenvalue. High = unstable optimiser input."""
    return float(np.linalg.cond(cov.values))