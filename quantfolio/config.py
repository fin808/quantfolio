from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

START_DATE = "2005-01-01"

UNIVERSE = {
    "equity_us":   ["SPY", "QQQ", "IWM"],
    "equity_intl": ["EFA", "EEM"],
    "bonds":       ["IEF", "TLT", "LQD", "HYG"],
    "commodities": ["GLD", "USO", "DBC"],
    "real_estate": ["VNQ"],
}
TICKERS = [t for group in UNIVERSE.values() for t in group]