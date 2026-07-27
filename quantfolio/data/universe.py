import pandas as pd


def interior_gaps(prices: pd.DataFrame) -> pd.Series:
    """NaNs occurring after a series has started — real holes, not inception."""
    return prices.apply(lambda c: c.loc[c.first_valid_index():].isna().sum())


def align(prices: pd.DataFrame, mode: str = "common", min_history: int = 252):
    """
    mode='common'  -> truncate to the window where every asset has data
    mode='longest' -> keep the full history, drop assets that start too late
    """
    if mode == "common":
        start = prices.apply(lambda c: c.first_valid_index()).max()
        out = prices.loc[start:].dropna(how="any")
    elif mode == "longest":
        starts = prices.apply(lambda c: c.first_valid_index())
        keep = starts[starts == starts.min()].index
        out = prices[keep].dropna(how="any")
    else:
        raise ValueError(f"unknown mode: {mode}")

    if len(out) < min_history:
        raise ValueError(f"only {len(out)} rows after alignment")
    return out