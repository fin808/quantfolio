import io
import urllib.request
import zipfile

import pandas as pd
import statsmodels.api as sm

from quantfolio.config import CACHE_DIR

FF_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"

DATASETS = {
    "ff3": "F-F_Research_Data_Factors_daily_CSV.zip",
    "ff5": "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip",
    "mom": "F-F_Momentum_Factor_daily_CSV.zip",
}


def _parse(text: str) -> pd.DataFrame:
    """Ken French CSVs have a text preamble, then a header row starting with
    a comma, then YYYYMMDD rows, then a copyright footer."""
    header, rows = None, []
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if header is None:
            if parts[0] == "" and len(parts) > 1 and any(parts[1:]):
                header = [p for p in parts[1:] if p]
            continue
        if len(parts[0]) == 8 and parts[0].isdigit():
            rows.append([parts[0]] + [float(x) for x in parts[1:len(header) + 1]])
        elif rows:
            break
    df = pd.DataFrame(rows, columns=["date"] + header)
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    df = df.set_index("date")
    return (df / 100.0).rename(columns={"Mkt-RF": "MKT"})


def load_factors(name: str = "ff3", refresh: bool = False) -> pd.DataFrame:
    path = CACHE_DIR / f"factors_{name}.parquet"
    if path.exists() and not refresh:
        return pd.read_parquet(path)
    with urllib.request.urlopen(FF_URL + DATASETS[name]) as r:
        z = zipfile.ZipFile(io.BytesIO(r.read()))
    df = _parse(z.read(z.namelist()[0]).decode("latin-1"))
    df.to_parquet(path)
    print(f"fetched {name}: {len(df)} rows, {df.index[0].date()} to {df.index[-1].date()}")
    return df


def factor_panel() -> pd.DataFrame:
    """FF5 + momentum + RF in one frame."""
    ff5 = load_factors("ff5")
    mom = load_factors("mom")
    mom.columns = ["MOM"]
    return ff5.join(mom, how="inner")


MODELS = {
    "CAPM": ["MKT"],
    "FF3": ["MKT", "SMB", "HML"],
    "Carhart 4": ["MKT", "SMB", "HML", "MOM"],
    "FF5": ["MKT", "SMB", "HML", "RMW", "CMA"],
    "FF5 + MOM": ["MKT", "SMB", "HML", "RMW", "CMA", "MOM"],
}


def regress(port: pd.Series, panel: pd.DataFrame, cols, lags: int = 5):
    """Excess-return regression with Newey-West standard errors."""
    df = pd.concat([port.rename("port"), panel], axis=1, join="inner").dropna()
    y = df["port"] - df["RF"]
    X = sm.add_constant(df[cols])
    return sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": lags})


def attribution(port: pd.Series, panel: pd.DataFrame) -> pd.DataFrame:
    rows = {}
    for name, cols in MODELS.items():
        res = regress(port, panel, cols)
        row = {"alpha_ann": round(res.params["const"] * 252, 4),
               "alpha_t": round(res.tvalues["const"], 2),
               "R2_adj": round(res.rsquared_adj, 3),
               "n": int(res.nobs)}
        for c in cols:
            row[c] = round(res.params[c], 3)
            row[f"t_{c}"] = round(res.tvalues[c], 1)
        rows[name] = row
    return pd.DataFrame(rows).T