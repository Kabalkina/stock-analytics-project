import os
import yaml
import pandas as pd

# Path to config.yaml 
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")

def _load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)

_config = _load_config()

# Static config values
START_DATE = pd.to_datetime(_config["START_DATE"])
END_DATE = pd.to_datetime(_config["END_DATE"])

TICKERS_LIST = _config["TICKERS_LIST"]
FRED_DEFAULTS = _config["FRED_DEFAULTS"]
MARKET_TICKERS = _config["MARKET_TICKERS"]
INITIAL_CAPITAL = _config["INITIAL_CAPITAL"]
TRANSACTION_FEE = _config["TRANSACTION_FEE"]
STOP_LOSS_PCT = _config["STOP_LOSS_PCT"]
TAKE_PROFIT_PCT = _config["TAKE_PROFIT_PCT"]

