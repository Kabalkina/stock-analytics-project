# automation.py
import os
import json
import logging
from pathlib import Path
import pandas as pd

from config_loader import START_DATE, END_DATE, TICKERS_LIST
from scripts.data import DataRepo
from scripts.model import Model
from scripts.simulation import StrategySimulation

# --- flags controlled by env (default: run everything) ---
LOAD_DATA = os.getenv("LOAD_DATA", "true").lower() == "true"
TRAIN_MODEL = os.getenv("TRAIN_MODEL", "true").lower() == "true"
RUN_SIMULATION = os.getenv("RUN_SIMULATION", "true").lower() == "true"

# --- logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("ci_automation")

DATA_DIRS = [Path("data"), Path("Data")]  # support either folder name
def existing_csv():
    for d in DATA_DIRS:
        f = d / "quarterly_data.csv"
        if f.exists():
            return f
    return None

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def main():
    log.info("Starting monthly pipeline | LOAD_DATA=%s | TRAIN_MODEL=%s | RUN_SIMULATION=%s",
             LOAD_DATA, TRAIN_MODEL, RUN_SIMULATION)

    # 1) DATA
    data = DataRepo()
    if LOAD_DATA:
        log.info("Building quarterly dataset from scratch")
        data.build_quarterly_dataset(TICKERS_LIST, START_DATE, END_DATE)
        data.save_dataset()
    else:
        csv = existing_csv()
        if csv is None:
            raise FileNotFoundError("No cached data file found: data/quarterly_data.csv or Data/quarterly_data.csv")
        log.info("Loading dataset from cached CSV: %s", csv)
        df = pd.read_csv(csv, parse_dates=["date"])
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        # Expect DataRepo to expose an attribute for downstream usage:
        data.quarterly_data = df
        log.info("Loaded %d rows", len(df))

    # 2) MODEL
    model = None
    if TRAIN_MODEL:
        log.info("Training model...")
        cached = Path("Data/rf_model_pred.csv")

        if cached.exists():
            log.info("Found existing rf_model_pred.csv → loading instead of retraining")
            df = pd.read_csv(cached, parse_dates=["date"])

            dummy = type("DummyModel", (), {})()
            dummy.df = df
            dummy.CLASSIFICATION_REPORT = {"note": "Skipped training – loaded from CSV"}
            dummy.CONFUSION_MATRIX = [[0, 0], [0, 0]]
            model = dummy
        else:
            # train from scratch
            model = Model(data)
            model.define_categories()
            model.get_dummies()
            model.handle_nans()
            model.split_data()
            model.train_model()
            model.make_predictions()
            model.save_data_for_simulation()
            log.info("Model trained & saved.")
    else:
        log.info("Skipping training → loading rf_model_pred.csv")
        cached = Path("Data/rf_model_pred.csv")
        if not cached.exists():
            raise FileNotFoundError("rf_model_pred.csv not found. You must run TRAIN_MODEL=true first.")
        df = pd.read_csv(cached, parse_dates=["date"])

        dummy = type("DummyModel", (), {})()
        dummy.df = df
        dummy.CLASSIFICATION_REPORT = {"note": "Skipped training – loaded from CSV"}
        dummy.CONFUSION_MATRIX = [[0, 0], [0, 0]]
        model = dummy

    # 3) SIMULATION
    if RUN_SIMULATION:
        log.info("Running backtest")
        sim = StrategySimulation()

        # Use model.df if model was trained
        if model is not None:
            df_for_sim = model.df
        else:
            # Otherwise load the cached rf_model_pred.csv
            cached = Path("data/rf_model_pred.csv")
            if not cached.exists():
                raise FileNotFoundError("rf_model_pred.csv not found. Set TRAIN_MODEL=true first.")
            df_for_sim = pd.read_csv(cached, parse_dates=["date"])
        history, trades, metrics = sim.backtest_fixed_capital(df_for_sim)

        out_dir = Path("data")
        ensure_dir(out_dir)

        # Save simple CSV/JSON outputs for traceability
        if history is not None and hasattr(history, "to_csv"):
            history.to_csv(out_dir / "backtest_history.csv", index=False)
        if trades is not None and hasattr(trades, "to_csv"):
            trades.to_csv(out_dir / "backtest_trades.csv", index=False)
        if metrics is not None:
            (out_dir / "backtest_metrics.json").write_text(json.dumps(metrics, indent=2))

        log.info("Backtest done | Metrics: %s", metrics)
    else:
        log.info("Skipping simulation (RUN_SIMULATION=false)")

    log.info("Pipeline finished successfully.")

if __name__ == "__main__":
    main()
