import numpy as np
import pandas as pd
import yfinance as yf
from quantfolio.config import CACHE_DIR, START_DATE


def _cache_path(ticker: str):
    return CACHE_DIR / f"{ticker}.parquet"


def fetch_prices(tickers, start=START_DATE, end=None, refresh=False) -> pd.DataFrame:
    """Total-return prices, one column per ticker. Cached to parquet."""
    frames = {}
    for t in tickers:
        path = _cache_path(t)
        if path.exists() and not refresh:
            s = pd.read_parquet(path)["adj_close"]
        else:
            df = yf.download(t, start=start, end=end, auto_adjust=True,
                             progress=False, multi_level_index=False)
            if df.empty:
                raise ValueError(f"No data returned for {t}")
            s = df["Close"].squeeze().rename("adj_close").dropna()
            s.to_frame().to_parquet(path)
            print(f"fetched {t}: {len(s)} rows, {s.index[0].date()} to {s.index[-1].date()}")
        frames[t] = s
    return pd.DataFrame(frames).sort_index()


def to_returns(prices: pd.DataFrame, log: bool = False) -> pd.DataFrame:
    r = np.log(prices).diff() if log else prices.pct_change()
    return r.dropna(how="all")