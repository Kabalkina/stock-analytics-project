
"""
Data extraction & feature engineering pipeline

This module keeps the *logic* of `data_extraction.ipynb` notebook, excluding visuals, 
and exposes clear functions used later by training/simulation:

- get_price_features(ticker, start, end, features=True)
- get_fundamental_features(ticker, asof=None)
- get_fred_macro_data(fred_series, start_date, end_date)
- get_macro_features(ticker_map, start_date, end_date)
- build_quarterly_dataset(tickers, start_date, end_date)

Outputs a quarterly dataset with price-, fundamental- and macro-features
ready for modeling. Fundamentals are computed once per ticker (latest/asof)
and joined across all quarters for that ticker
"""

from __future__ import annotations

import os
import math
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from pandas_datareader import data as pdr
from pathlib import Path

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("data_pipeline")


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def _to_quarter_end_idx(df: pd.DataFrame, date_col: str = None) -> pd.DataFrame:
    """Ensure a DatetimeIndex at quarter-end for resampling/joins."""
    d = df.copy()
    if date_col and date_col in d.columns:
        d[date_col] = pd.to_datetime(d[date_col])
        d = d.set_index(date_col)
    d.index = pd.to_datetime(d.index)
    # Align to calendar quarter-end
    d = d.groupby(pd.Grouper(freq="Q")).last()
    d.index.name = "date"
    return d


def _safe_div(a, b):
    return np.where(np.abs(b) > 0, a / b, np.nan)


# -----------------------------------------------------------------------------
# Price features (daily → quarterly)
# -----------------------------------------------------------------------------
def get_price_features(ticker: str, start: str, end: Optional[str], features: bool = True) -> pd.DataFrame:
    """
    Download daily OHLCV with yfinance, compute derived fields, resample to **quarter-end**.

    Features engineered (quarterly):
    - ln_volume (median within quarter)
    - median daily / 5d / 10d / 21d growth (Close)
    - spreads: open-close, high-low (absolute & relative, medians within quarter)
    - vol_10d (rolling std of daily returns → quarterly median)
    - quarterly close
    - return_1q / return_2q / return_3q (on quarterly close)
    - vol_3q (rolling std of quarterly returns over 3 qtrs)
    - momentum_3q (Close pct_change over 3 qtrs)
    - qma_2 / qma_4 / qma_8 (quarterly moving averages of Close)
    """
    yf.pdr_override()  # just in case

    try:
        raw = yf.Ticker(ticker).history(ticker, start=start, end=end, interval="1d")
        if raw.empty:
            logger.warning(f"No price data for {ticker}")
            return pd.DataFrame()
    except Exception as e:
        logger.exception(f"yfinance failed for {ticker}: {e}")
        return pd.DataFrame()

    # Use Adjusted Close if available; else Close
    if "Adj Close" in raw.columns:
        raw = raw.rename(columns={"Adj Close": "AdjClose"})
        close = raw["AdjClose"].copy()
    else:
        close = raw["Close"].copy()

    df = raw.copy()

    df["ln_volume"] = np.log(df['Volume'].replace(0, np.nan))

    # Daily returns and horizon returns
    daily_ret = close.pct_change()
    ret_5 = close.pct_change(5) # weekly
    ret_10 = close.pct_change(10) # biweekly
    ret_21 = close.pct_change(21) # monthly

    # Spreads
    spread_oc = df['Open'] - df['Close']
    spread_hl = df['High'] - df['Low']

    # Relative spreads (scaled by Close price)
    rel_spread_oc = df['spread_oc'] / df['Close']
    rel_spread_hl = df['spread_hl'] / df['Close']

    # Rolling volatility for short-term risk
    vol_10d = daily_ret.rolling(10).std()

    # Assemble daily frame for resample
    daily = pd.DataFrame({
        "close": close,
        "ln_volume": df.get("ln_volume"),
        "daily_ret": daily_ret,
        "ret_5": ret_5,
        "ret_10": ret_10,
        "ret_21": ret_21,
        "spread_oc": spread_oc,
        "spread_hl": spread_hl,
        "rel_spread_oc": rel_spread_oc,
        "rel_spread_hl": rel_spread_hl,
        "vol_10d": vol_10d,
    }).dropna(how="all")

    # Resample to quarter-end aggregations
    agg = {
        "close": "last",
        "ln_volume": "median",
        "daily_ret": "median",
        "ret_5": "median",
        "ret_10": "median",
        "ret_21": "median",
        "spread_oc": "median",
        "spread_hl": "median",
        "rel_spread_oc": "median",
        "rel_spread_hl": "median",
        "vol_10d": "median",
    }
    q = daily.resample("QE").agg(agg)

    # Quarterly derived features
    q["return_1q"] = q["close"].pct_change(1)
    q["return_2q"] = q["close"].pct_change(2)
    q["return_3q"] = q["close"].pct_change(3)

    # Volatility & momentum
    q["vol_3q"] = q["return_1q"].rolling(3).std()
    q["momentum_3q"] = q["close"].pct_change(3)

    # Quarterly moving averages
    q["qma_2"] = q["close"].rolling(2).mean()
    q["qma_4"] = q["close"].rolling(4).mean()
    q["qma_8"] = q["close"].rolling(8).mean()

    # Convert index to datetime, remove timezone
    q.index = pd.to_datetime(q.index).tz_localize(None)

    q = q.dropna(subset=["close"]).reset_index().rename(columns={"index": "date"})
    q["ticker"] = ticker
    q["year"] = q["date"].dt.year
    q["quarter"] = q["date"].dt.quarter
    return q


# -----------------------------------------------------------------------------
# Fundamentals (latest/as-of)
# -----------------------------------------------------------------------------
def _get_latest_col(df: pd.DataFrame) -> Optional[pd.Series]:
    if df is None or df.empty:
        return None
    # yfinance quarterly frames: columns are dates; take the latest column
    latest = df.iloc[:, 0] if df.columns.size else None
    return latest


def get_fundamental_features(ticker: str, asof: Optional[str] = None) -> pd.DataFrame:
    """
    Pulls *latest* fundamentals for a ticker via yfinance and computes ratios.
    Returned as a single-row DataFrame (to be joined to each quarterly row for that ticker).

    Includes (best-effort, many are optional in Yahoo data):
    - marketCap, beta, trailingPE, forwardPE, priceToBook, dividendYield
    - debt_to_equity, long_term_debt_to_equity
    - current_ratio, cash_ratio, working_capital, working_capital_ratio
    - retained_earnings_to_assets
    - operating_cf_margin, free_cf_margin
    - pct_held_by_inst, institutional_holders_count
    - esg_env, esg_soc, esg_gov
    - recent_rating_changes, strong_to_total_reco_ratio
    - days_to_next_earnings
    - companyName, sector, industry, country, fullTimeEmployees
    """
    t = yf.Ticker(ticker)

    # Basic profile/info
    info = {}
    try:
        info = t.info or {}
    except Exception:
        try:
            info = t.get_info() or {}
        except Exception:
            info = {}

    # Financial statements
    bs_q = None
    is_q = None
    cf_q = None
    try:
        bs_q = t.quarterly_balance_sheet
    except Exception:
        pass
    try:
        is_q = t.quarterly_financials
    except Exception:
        pass
    try:
        cf_q = t.quarterly_cashflow
    except Exception:
        pass

    bs = _get_latest_col(bs_q)
    isl = _get_latest_col(is_q)
    cfl = _get_latest_col(cf_q)

    # Ratios (guard against missing keys)
    def g(s, k):
        try:
            return float(s.get(k, np.nan)) if s is not None else np.nan
        except Exception:
            return np.nan

    total_assets = g(bs, "Total Assets")
    total_liab = g(bs, "Total Liab")
    equity = g(bs, "Total Stockholder Equity")
    current_assets = g(bs, "Total Current Assets")
    current_liab = g(bs, "Total Current Liabilities")
    cash = g(bs, "Cash") + g(bs, "Cash And Cash Equivalents")
    long_term_debt = g(bs, "Long Term Debt")
    retained_earnings = g(bs, "Retained Earnings")

    working_capital = current_assets - current_liab if not np.isnan(current_assets) and not np.isnan(current_liab) else np.nan
    working_capital_ratio = _safe_div(current_assets, current_liab)
    debt_to_equity = _safe_div(total_liab, equity)
    ltd_to_equity = _safe_div(long_term_debt, equity)
    cash_ratio = _safe_div(cash, current_liab)
    re_to_assets = _safe_div(retained_earnings, total_assets)

    total_revenue = g(isl, "Total Revenue")
    operating_cf = g(cfl, "Total Cash From Operating Activities")
    free_cf = g(cfl, "Free Cash Flow")
    op_cf_margin = _safe_div(operating_cf, total_revenue)
    free_cf_margin = _safe_div(free_cf, total_revenue)

    # Holders / ESG / Ratings
    try:
        inst = t.institutional_holders or pd.DataFrame()
    except Exception:
        inst = pd.DataFrame()
    pct_held_by_inst = float(inst.get("% Out", pd.Series(dtype=float)).fillna(0).sum()) if not inst.empty else np.nan
    institutional_holders_count = int(len(inst)) if not inst.empty else 0

    try:
        sustain = t.sustainability or pd.DataFrame()
    except Exception:
        sustain = pd.DataFrame()
    esg_env = float(sustain.get("environmentScore", pd.Series([np.nan])).iloc[0]) if not sustain.empty else np.nan
    esg_soc = float(sustain.get("socialScore", pd.Series([np.nan])).iloc[0]) if not sustain.empty else np.nan
    esg_gov = float(sustain.get("governanceScore", pd.Series([np.nan])).iloc[0]) if not sustain.empty else np.nan

    try:
        rec = t.recommendations or pd.DataFrame()
    except Exception:
        rec = pd.DataFrame()
    recent_rating_changes = int(rec[rec.index >= (pd.Timestamp.today() - pd.Timedelta(days=180))].shape[0]) if not rec.empty else 0
    strong_ratio = np.nan
    if not rec.empty and "To Grade" in rec:
        to_grade = rec["To Grade"].astype(str).str.lower()
        strong_buys = to_grade.str.contains("strong buy").sum()
        strong_ratio = strong_buys / max(len(to_grade), 1)

    # Earnings calendar → days to next earnings
    days_to_next_earnings = np.nan
    try:
        cal = t.calendar
        if cal is not None and not cal.empty:
            # Try common keys
            if "Earnings Date" in cal.index:
                ed = cal.loc["Earnings Date"].max()
                if pd.notna(ed):
                    days_to_next_earnings = (pd.to_datetime(ed) - pd.Timestamp.today()).days
    except Exception:
        pass

    # Assemble row
    row = {
        "ticker": ticker,
        "companyName": info.get("longName") or info.get("shortName"),
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "country": info.get("country"),
        "fullTimeEmployees": info.get("fullTimeEmployees"),
        "marketCap": info.get("marketCap"),
        "beta": info.get("beta"),
        "trailingPE": info.get("trailingPE"),
        "forwardPE": info.get("forwardPE"),
        "trailing_PEG": info.get("trailingPegRatio") or info.get("pegRatio"),
        "priceToBook": info.get("priceToBook"),
        "dividendYield": info.get("dividendYield"),
        "debt_to_equity": float(debt_to_equity) if not isinstance(debt_to_equity, np.ndarray) else float(debt_to_equity[0]),
        "current_ratio": float(working_capital_ratio) if not isinstance(working_capital_ratio, np.ndarray) else float(working_capital_ratio[0]),
        "cash_ratio": float(cash_ratio) if not isinstance(cash_ratio, np.ndarray) else float(cash_ratio[0]),
        "working_capital": working_capital,
        "working_capital_ratio": float(working_capital_ratio) if not isinstance(working_capital_ratio, np.ndarray) else float(working_capital_ratio[0]),
        "retained_earnings_to_assets": float(re_to_assets) if not isinstance(re_to_assets, np.ndarray) else float(re_to_assets[0]),
        "long_term_debt_to_equity": float(ltd_to_equity) if not isinstance(ltd_to_equity, np.ndarray) else float(ltd_to_equity[0]),
        "operating_cf_margin": float(op_cf_margin) if not isinstance(op_cf_margin, np.ndarray) else float(op_cf_margin[0]),
        "free_cf_margin": float(free_cf_margin) if not isinstance(free_cf_margin, np.ndarray) else float(free_cf_margin[0]),
        "pct_held_by_inst": pct_held_by_inst,
        "institutional_holders_count": institutional_holders_count,
        "esg_env": esg_env,
        "esg_soc": esg_soc,
        "esg_gov": esg_gov,
        "recent_rating_changes": recent_rating_changes,
        "strong_to_total_reco_ratio": strong_ratio,
        "days_to_next_earnings": days_to_next_earnings,
    }

    return pd.DataFrame([row])


# -----------------------------------------------------------------------------
# Macro features (FRED + market)
# -----------------------------------------------------------------------------
FRED_DEFAULTS = {
    # United States
    "gdp_us": "GDPC1",          # Real GDP (Billions Chained 2017$), Quarterly
    "cpi_us": "CPIAUCSL",       # CPI All Urban Consumers, Monthly → resampled Q
    "unemployment_us": "UNRATE", # Unemployment Rate, Monthly → Q
    "interest_us": "FEDFUNDS",   # Effective Federal Funds Rate, Monthly → Q
    # Germany / EU proxies (available at FRED)
    "gdp_de": "CLVMNACSCAB1GQDE",     # Real GDP, Germany, Quarterly
    "cpi_de": "DEUCPIALLMINMEI",      # CPI, Germany, Monthly → Q
    "interest_eu": "ECBDFR",           # ECB Deposit Facility Rate, Monthly → Q
}

MARKET_TICKERS = {
    "sp500": "^GSPC",
    "vix": "^VIX",
    "dax": "^GDAXI",
    "spy": "SPY",
    "gld": "GLD",
    "vgk": "VGK",
    "xlv": "XLV",
    "xlk": "XLK",
    "xlf": "XLF",
    "efa": "EFA",
    "eurusd": "EURUSD=X",
    "usdx": "DX-Y.NYB",
}


def get_fred_macro_data(fred_series: Dict[str, str], start_date: str, end_date: Optional[str]) -> pd.DataFrame:
    """Download FRED series and resample to quarterly end. Adds _yoy and _qoq columns."""
    frames = []
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date) if end_date else pd.Timestamp.today()

    for col, sid in fred_series.items():
        try:
            s = pdr.DataReader(sid, "fred", start, end)
            s.columns = [col]
            s_q = _to_quarter_end_idx(s)
            # Derivatives
            s_q[f"{col}_qoq"] = s_q[col].pct_change(1)
            s_q[f"{col}_yoy"] = s_q[col].pct_change(4)
            frames.append(s_q)
        except Exception as e:
            logger.warning(f"FRED fetch failed for {col}/{sid}: {e}")
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, axis=1)
    out = out.reset_index()  # with 'date'
    return out


def get_macro_features(ticker_map: Dict[str, str], start_date: str, end_date: Optional[str]) -> pd.DataFrame:
    """Download market indices/ETFs with yfinance and compute _qoq/_yoy on quarterly closes."""
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date) if end_date else None

    data = {}
    for col, tkr in ticker_map.items():
        try:
            px = yf.download(tkr, start=start, end=end, interval="1d", progress=False)
            if px.empty:
                logger.warning(f"No market data for {tkr}")
                continue
            close = px.get("Adj Close", px.get("Close")).rename(col)
            q = close.resample("Q").last().to_frame()
            q[f"{col}_qoq"] = q[col].pct_change(1)
            q[f"{col}_yoy"] = q[col].pct_change(4)
            data[col] = q
        except Exception as e:
            logger.warning(f"Market fetch failed for {tkr}: {e}")

    if not data:
        return pd.DataFrame()

    # Merge all on date index
    out = None
    for df in data.values():
        out = df if out is None else out.join(df, how="outer")
    out = out.reset_index().rename(columns={"index": "date"})
    return out


# -----------------------------------------------------------------------------
# Dataset builder
# -----------------------------------------------------------------------------
def build_quarterly_dataset(tickers: List[str], start_date: str, end_date: Optional[str]) -> pd.DataFrame:
    """
    Build the full quarterly dataset:
      1) per-ticker price features (quarterly)
      2) per-ticker fundamentals (latest/as-of) broadcast across quarters
      3) macro features (FRED + market indices/ETFs) merged on date
    """
    # 1) Price features for each ticker
    parts = []
    for t in tickers:
        logger.info(f"Pricing/features for {t}")
        q = get_price_features(t, start=start_date, end=end_date, features=True)
        if not q.empty:
            parts.append(q)
    if not parts:
        logger.error("No price features produced.")
        return pd.DataFrame()

    prices = pd.concat(parts, ignore_index=True)

    # 2) Fundamentals per ticker (latest) → broadcast across ticker's quarters
    fund_rows = []
    for t in tickers:
        logger.info(f"Fundamentals for {t}")
        row = get_fundamental_features(t)
        if row is None or row.empty:
            continue
        fund_rows.append(row)
    fundamentals = pd.concat(fund_rows, ignore_index=True) if fund_rows else pd.DataFrame()

    if not fundamentals.empty:
        prices = prices.merge(fundamentals, on="ticker", how="left")

    # 3) Macro features
    fred = get_fred_macro_data(FRED_DEFAULTS, start_date, end_date)
    macro_mkt = get_macro_features(MARKET_TICKERS, start_date, end_date)

    out = prices.copy()
    if not fred.empty:
        out = out.merge(fred, on="date", how="left")
    if not macro_mkt.empty:
        out = out.merge(macro_mkt, on="date", how="left")

    # Final ordering
    cols_first = [
        "ticker", "date", "year", "quarter", "close",
        "return_1q", "return_2q", "return_3q", "vol_3q", "momentum_3q",
        "qma_2", "qma_4", "qma_8",
    ]
    # put available first, then the rest
    existing_first = [c for c in cols_first if c in out.columns]
    rest = [c for c in out.columns if c not in existing_first]
    out = out[existing_first + rest]

    return out


# -----------------------------------------------------------------------------
# CLI entry (optional local run)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build quarterly dataset")
    parser.add_argument("--tickers_csv", type=str, default="data/tickers.csv", help="CSV with column 'ticker'")
    parser.add_argument("--start", type=str, default="2000-01-01")
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--out", type=str, default="data/processed_data.csv")
    args = parser.parse_args()

    # Load tickers list
    if Path(args.tickers_csv).exists():
        tl = pd.read_csv(args.tickers_csv)
        if "ticker" in tl.columns:
            tickers = tl["ticker"].dropna().astype(str).unique().tolist()
        else:
            raise ValueError("tickers.csv must have a 'ticker' column")
    else:
        logger.warning("tickers.csv not found; using example tickers")
        tickers = ["AAPL", "MSFT", "JNJ", "PFE", "SAP.DE", "IFX.DE"]

    ds = build_quarterly_dataset(tickers, args.start, args.end)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    ds.to_csv(args.out, index=False)
    logger.info(f"Saved dataset → {args.out}")
