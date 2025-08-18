#run script

from scripts.1_data import DataRepository
from scripts.2_model import TransformData
from scripts.3_simulation import StrategySimulation

import pandas as pd
import warnings
import os

from datetime import datetime  # Import the datetime module

# -----------------------------------------------------------------------------
# CLI entry (optional local run)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    #import argparse

    #parser = argparse.ArgumentParser(description="Build quarterly dataset")
    #parser.add_argument("--tickers_csv", type=str, default="data/tickers.csv", help="CSV with column 'ticker'")
    #parser.add_argument("--start", type=str, default="2000-01-01")
    #parser.add_argument("--end", type=str, default=None)
    #parser.add_argument("--out", type=str, default="data/processed_data.csv")
    #args = parser.parse_args()

    # Load tickers list
    # if Path(args.tickers_csv).exists():
    #     tl = pd.read_csv(args.tickers_csv)
    #     if "ticker" in tl.columns:
    #         tickers = tl["ticker"].dropna().astype(str).unique().tolist()
    #     else:
    #         raise ValueError("tickers.csv must have a 'ticker' column")
    # else:
    #     logger.warning("tickers.csv not found; using example tickers")
    #     tickers = ["AAPL", "MSFT", "JNJ", "PFE", "SAP.DE", "IFX.DE"]

    ds = build_quarterly_dataset(TICKERS_LIST, START_DATE, END_DATE)
    #Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    #ds.to_csv(out, index=False)
    logger.info(f"Saved dataset → {out}")