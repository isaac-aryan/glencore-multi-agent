"""
data.py — Download, cache, clean, and align all series.

All downstream notebooks import from here. Nothing hardcoded:
tickers and paths come from config.yaml.

Usage:
    from glencore_quant.data import load_glencore, load_commodities, load_aligned_panel
"""

from pathlib import Path
import yaml
import numpy as np
import pandas as pd
import yfinance as yf


# ─── Config ───────────────────────────────────────────────────────────────────

def load_config(config_path: str = None) -> dict:
    """Load project config from config.yaml"""
    if config_path is None:
        config_path = Path(__file__).parent.parent.parent / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


CFG = load_config()
RAW_DIR = Path(CFG["paths"]["raw"])
INTERIM_DIR = Path(CFG["paths"]["interim"])


# ─── Core download helper ──────────────────────────────────────────────────────

def _download_or_load(
    ticker: str,
    name: str,
    start: str,
    force_refresh: bool = False,
    auto_adjust: bool = False,
) -> pd.DataFrame:
    """
    Download ticker from yfinance and cache to data/raw/{name}.csv.
    On subsequent calls, loads from cache unless force_refresh=True.

    auto_adjust=False so we get both Close and Adj Close for Glencore.
    auto_adjust=True for commodities (only one price column needed).
    """
    cache_path = RAW_DIR / f"{name}.csv"

    if cache_path.exists() and not force_refresh:
        print(f"  Loading {name} from cache: {cache_path}")
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        return df

    print(f"  Downloading {name} ({ticker}) from yfinance...")
    df = yf.download(
        ticker,
        start=start,
        auto_adjust=auto_adjust,
        progress=False,
    )

    if df.empty:
        raise ValueError(f"No data returned for {ticker}. Check ticker or internet connection.")

    # Flatten multi-level columns (yfinance can return these)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.index.name = "Date"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path)
    print(f"  Cached to {cache_path}")
    return df


# ─── Glencore ─────────────────────────────────────────────────────────────────

def load_glencore(force_refresh: bool = False) -> pd.DataFrame:
    """
    Load GLEN.L OHLCV with both raw Close and Adj Close.

    Returns a DataFrame with columns:
        open, high, low, close, adj_close, volume
        log_return  (computed from adj_close)
        close_gbp   (close / 100 — pounds, not pence)
        adj_close_gbp

    Note: raw prices are in GBX (pence). See 'currency' in config.yaml.
    """
    cfg = CFG["glencore"]
    df = _download_or_load(
        ticker=cfg["ticker"],
        name="glen_raw",
        start=cfg["start_date"],
        force_refresh=force_refresh,
        auto_adjust=False,  # keep both Close and Adj Close
    )

    # Standardise column names to lowercase
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]

    # GBX → GBP convenience columns (divide by 100)
    df["close_gbp"] = df["close"] / 100
    df["adj_close_gbp"] = df["adj_close"] / 100

    # Log returns from adjusted price (the series everything else will use)
    df["log_return"] = np.log(df["adj_close"] / df["adj_close"].shift(1))

    # Drop the very first row (NaN return)
    df = df.dropna(subset=["log_return"])

    print(f"\nGlencore loaded: {len(df)} rows, {df.index[0].date()} → {df.index[-1].date()}")
    return df


def load_dividends() -> pd.Series:
    """Return the dividend history for GLEN.L as a Series indexed by ex-date."""
    t = yf.Ticker(CFG["glencore"]["ticker"])
    return t.dividends


# ─── Commodities ──────────────────────────────────────────────────────────────

def load_commodities(force_refresh: bool = False) -> pd.DataFrame:
    """
    Load commodity and macro series. Returns a DataFrame with one
    column per series (close prices only), and a log_return column
    for each (prefixed: e.g. 'copper_log_ret').

    Known data quality issues documented inline.
    """
    all_tickers = {
        **CFG["commodities"],
        **CFG["macro"],
    }

    frames = {}
    start = CFG["glencore"]["start_date"]

    for name, ticker in all_tickers.items():
        try:
            df = _download_or_load(
                ticker=ticker, name=name, start=start,
                force_refresh=force_refresh, auto_adjust=True
            )
            close = df["Close"].rename(name)
            # Flag if coverage is sparse (e.g. zinc futures)
            pct_missing = close.isna().mean() * 100
            if pct_missing > 20:
                print(f"  ⚠  {name}: {pct_missing:.1f}% missing — may be unreliable")
            frames[name] = close
        except Exception as e:
            print(f"  ✗  {name} ({ticker}): {e}")

    panel = pd.DataFrame(frames)

    # Add log returns for each commodity
    for col in frames.keys():
        panel[f"{col}_log_ret"] = np.log(panel[col] / panel[col].shift(1))

    print(f"\nCommodity panel: {panel.shape}, {panel.index[0].date()} → {panel.index[-1].date()}")
    return panel


# ─── Aligned panel ────────────────────────────────────────────────────────────

def load_aligned_panel(
    fill_method: str = "ffill",
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Return a single, date-aligned panel of Glencore + all drivers.

    fill_method: 'ffill' (forward-fill holiday gaps) or 'drop' (drop any row
                 with a missing value). 'ffill' is standard for daily data.
                 Document your choice — it affects downstream tests.

    Saves to data/interim/aligned_panel.csv.
    """
    interim_path = INTERIM_DIR / "aligned_panel.csv"

    if interim_path.exists() and not force_refresh:
        print(f"Loading aligned panel from {interim_path}")
        return pd.read_csv(interim_path, index_col=0, parse_dates=True)

    print("Building aligned panel...")
    glen = load_glencore(force_refresh=force_refresh)
    comms = load_commodities(force_refresh=force_refresh)

    # Use Glencore's trading dates as the master index (LSE calendar)
    panel = glen.copy()

    # Reindex commodities to LSE dates, apply fill method
    comms_aligned = comms.reindex(panel.index)

    if fill_method == "ffill":
        comms_aligned = comms_aligned.ffill()
    elif fill_method == "drop":
        pass  # NaNs will be present — caller can dropna()

    panel = pd.concat([panel, comms_aligned], axis=1)

    # Drop any row still missing Glencore data (shouldn't happen but be safe)
    panel = panel.dropna(subset=["adj_close"])

    # Save
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    panel.to_csv(interim_path)
    print(f"Saved aligned panel: {panel.shape} → {interim_path}")
    print(f"Remaining NaN counts:\n{panel.isna().sum()[panel.isna().sum() > 0]}")

    return panel