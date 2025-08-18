
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

from matplotlib import ticker
import numpy as np
import pandas as pd
import yfinance as yf
from pandas_datareader import data as pdr
from pathlib import Path


START_DATE = pd.to_datetime("2000-01-01")
END_DATE = pd.Timestamp.today().strftime('%Y-%m-%d')

TICKERS_LIST = ['AAPL', 'MSFT', 'NVDA', 'ORCL', 'IBM', 'GOOGL', 'AMZN', 'META',
 'CSCO', 'INTU', 'JNJ', 'PFE', 'MRK', 'ABBV', 'UNH', 'LLY', 'AMGN', 'BMY',
 'KO', 'PG', 'PEP', 'JPM', 'BAC', 'GS', 'MS', 'XOM', 'CVX', 'CAT', 'GE',
 'MMM', 'SAP.DE', 'IFX.DE', 'FME.DE', 'SHL.DE', 'BAS.DE', 'BAYN.DE', 'SIE.DE',
 'ALV.DE', 'DBK.DE', 'HSBA.L', 'BP.L', 'ULVR.L', 'AIR.PA', 'OR.PA', 'ASML.AS',
 'NOVN.SW', 'ROG.SW', 'UBSG.SW', '7203.T', '6758.T', '9432.T', '9984.T', '7267.T', 
 '6861.T', '6501.T']

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
    "xlv": "XLV", # healthcare
    "xlk": "XLK", # technology
    "xlf": "XLF", # financials
    "efa": "EFA", # developed markets
    "eurusd": "EURUSD=X", # euro to usd
    "usdx": "DX-Y.NYB", # dollar index
}



# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("data_pipeline")

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
    try:
        tk = yf.Ticker(ticker)
        info = tk.get_info() or {}
        bs = tk.get_balance_sheet()
        cf = tk.get_cashflow()
        holders = tk.get_institutional_holders()
        sustainability = tk.get_sustainability()
        recommendations = tk.get_recommendations()
        calendar = tk.get_calendar()

        # Ensure to take the latest available data from balance sheet
        def safe_get_latest(bs_df, row_name):
            if bs_df is not None and row_name in bs_df.index:
                series = bs_df.loc[row_name].dropna()
                if not series.empty:
                    return series.iloc[0]
            return np.nan
 
        # Fundamental ratios
        
        # BS parameters
        total_assets = safe_get_latest(bs, 'TotalAssets')
        total_liabilities = safe_get_latest(bs, 'TotalLiabilitiesNetMinorityInterest')
        total_equity = safe_get_latest(bs, 'StockholdersEquity')
        long_term_debt = safe_get_latest(bs, 'LongTermDebt')
        current_assets = safe_get_latest(bs, 'CurrentAssets')
        current_liabilities = safe_get_latest(bs, 'CurrentLiabilities')
        cash_equivalents = safe_get_latest(bs, 'CashAndCashEquivalents')
        retained_earnings = safe_get_latest(bs, 'RetainedEarnings')
        working_capital = safe_get_latest(bs, 'WorkingCapital')

        debt_to_equity = total_liabilities / total_equity if total_equity else np.nan
        current_ratio = current_assets / current_liabilities if current_liabilities else np.nan
        cash_ratio = cash_equivalents / current_liabilities if current_liabilities else np.nan
        working_capital_ratio = working_capital / total_assets if total_assets else np.nan
        retained_earnings_to_assets = retained_earnings / total_assets if total_assets else np.nan
        long_term_debt_to_equity = long_term_debt / total_equity if total_equity else np.nan
    
        op_cf = cf.loc['OperatingCashFlow'].iloc[0] if cf is not None and 'OperatingCashFlow' in cf.index else None
        revenue = info.get('totalRevenue')
        operating_cf_margin = op_cf / revenue if op_cf and revenue else np.nan
        

        free_cf = cf.loc['FreeCashFlow'].iloc[0] if cf is not None and 'FreeCashFlow' in cf.index else None
        free_cf_margin = free_cf / revenue if free_cf and revenue else np.nan

        # Institutional holders
        pct_held = holders['pctHeld'].mean() if holders is not None and not holders.empty else np.nan
        inst_count = len(holders) if holders is not None else np.nan

        # ESG scores
        esg_env = sustainability.loc['environmentScore'].iloc[0] if sustainability is not None and 'environmentScore' in sustainability.index else np.nan
        esg_soc = sustainability.loc['socialScore'].iloc[0] if sustainability is not None and 'socialScore' in sustainability.index else np.nan
        esg_gov = sustainability.loc['governanceScore'].iloc[0] if sustainability is not None and 'governanceScore' in sustainability.index else np.nan

        # Recommendations count last 90 days
        recent_reco = 0
        if recommendations is not None and not recommendations.empty:
            recent_months = recommendations[recommendations['period'].isin(['0m', '-1m', '-2m', '-3m'])]
            recent_reco = recent_months[['strongBuy', 'buy', 'hold', 'sell', 'strongSell']].sum().sum()
            strong_to_total_ratio = recent_months['strongBuy'].sum() / recent_reco if recent_reco else np.nan

        # Days to next earnings
        days_to_earnings = np.nan
        if calendar and isinstance(calendar, dict) and 'Earnings Date' in calendar:
            earnings_dates = calendar['Earnings Date']
            if isinstance(earnings_dates, list) and len(earnings_dates) > 0:
                earnings_date = earnings_dates[0]
                if isinstance(earnings_date, (datetime.date, datetime.datetime)):
                    earnings_datetime = pd.to_datetime(earnings_date)
                    days_to_earnings = (earnings_datetime - pd.to_datetime(END_DATE)).days
        
        # Dividend stability: std of yearly dividend sums / mean dividend sums
        actions = tk.get_actions()
        dividend_stability = np.nan
        if actions is not None and 'Dividends' in actions.columns:
            yearly_divs = actions['Dividends'].resample('YE').sum()
            if len(yearly_divs) > 1 and yearly_divs.mean() != 0:
                dividend_stability = yearly_divs.std() / yearly_divs.mean()

        # Assemble row
        row = {
            "ticker": ticker,
            "companyName": info.get("shortName", np.nan),
            "sector": info.get("sector", np.nan),
            "fullTimeEmployees": info.get("fullTimeEmployees", np.nan),
            "marketCap": info.get("marketCap", np.nan),
            "beta": info.get("beta", np.nan),
            "trailingPE": info.get("trailingPE", np.nan),
            "forwardPE": info.get("forwardPE", np.nan),
            "trailing_PEG": info.get("trailingPegRatio", np.nan) or info.get("pegRatio", np.nan),
            "priceToBook": info.get("priceToBook", np.nan),
            "dividendYield": info.get("dividendYield", np.nan),
            "debt_to_equity": debt_to_equity,
            "current_ratio": working_capital_ratio,
            "cash_ratio": cash_ratio,
            "working_capital": working_capital,
            "working_capital_ratio": working_capital_ratio,
            "retained_earnings_to_assets": retained_earnings_to_assets,
            "long_term_debt_to_equity": long_term_debt_to_equity,
            "operating_cf_margin": operating_cf_margin,
            "free_cf_margin": free_cf_margin,
            "institutional_holders_count": inst_count,
            "esg_env": esg_env,
            "esg_soc": esg_soc,
            "esg_gov": esg_gov,
            "recent_rating_changes": recent_reco,
            "strong_to_total_reco_ratio": strong_to_total_ratio,
            "days_to_next_earnings": days_to_earnings,
        }

        return pd.DataFrame([row])
    
    except Exception as e:
        print(f"[FUNDAMENTAL] Error for {ticker}: {e}")
        return {}



# -----------------------------------------------------------------------------
# Macro features (FRED + market)
# -----------------------------------------------------------------------------
def get_fred_macro_data(fred_series: Dict[str, str], start_date: str, end_date: Optional[str]) -> pd.DataFrame:
    """
    Fetch FRED macroeconomic data, compute YoY and QoQ changes,
    and return a single merged DataFrame resampled to quarter-end.

    Parameters:
    - fred_series: dict of {friendly_name: FRED_series_code}
    - start_date, end_date: date strings in 'YYYY-MM-DD' format

    Returns:
    - macro_df: DataFrame with macro features per quarter
    """
    frames = []

    for name, code in fred_series.items():
        try:
            df = pdr.DataReader(code, "fred", start_date, end_date)
            # Rename column to friendly name
            df.columns = [name]
            
            # YoY and QoQ percentage changes
            df[name + '_yoy'] = df[name].pct_change(4)
            df[name + '_qoq'] = df[name].pct_change(1)
            frames.append(df)
        except Exception as e:
            logger.warning(f"FRED fetch failed for {name}/{code}: {e}")
    if not frames:
        return pd.DataFrame()
    
    out = pd.concat(frames, axis=1).resample('QE').last()
    out = out.reset_index().rename(columns={'DATE': 'date'})
    return out


def get_macro_features(ticker_map: Dict[str, str], start_date: str, end_date: Optional[str]) -> pd.DataFrame:
    """
    Download macro ETF/index data from yfinance, resample quarterly,
    and compute simple moving average features.
    
    Parameters:
    - ticker_map: dict of {feature_name: yfinance_ticker}
    - start_date: start date string (e.g. '2000-01-01')
    - end_date: end date string
    
    Returns:
    - DataFrame with quarterly macro features
    """

    data = {}
    for name, tkr in ticker_map.items():
        try:
            px = yf.Ticker(tkr).history(start=start_date, end=end_date, interval="1d")
            if px.empty:
                logger.warning(f"No market data for {tkr}")
                continue

            price_col = 'Adj Close' if 'Adj Close' in data.columns else 'Close'
            df = data[[price_col]].rename(columns={price_col: 'Close'})

            df = df.resample("QE").last().to_frame()
            df.index = pd.to_datetime(df.index).tz_localize(None)
            df.reset_index(inplace=True)

            # YoY and QoQ percentage changes
            df[f"{name}_qoq"] = df[name].pct_change(1)
            df[f"{name}_yoy"] = df[name].pct_change(4)
            data[name] = df
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

    # 4) Buy signal (example logic)
    buy_signal_1 = (
        (out['qma_2'] > out['qma_4']) &
        (out['return_2q'] > out['sp500_qoq'])
        )

    # Catch value+trend buys (reasonable valuation, positive macro backdrop)
    buy_signal_2 = (
        (out['qma_2'] > out['qma_4']) &
        (out['momentum_3q'] > 0) &
        (out['trailingPE'] < 30) &
        (out['interest_us_qoq'] < 0.5)
    )

    out['buy_signal'] = buy_signal_1 | buy_signal_2

    return out

