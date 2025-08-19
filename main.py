#run script


from scripts.data import DataRepo
from scripts.model import Model
from scripts.simulation import StrategySimulation

import pandas as pd
import warnings
import os

from datetime import datetime  # Import the datetime module

from config_loader import START_DATE, END_DATE, TICKERS_LIST

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("main_pipeline")

def main():
    
    # 1. load data
    data = DataRepo()
    data.build_quarterly_dataset(TICKERS_LIST, START_DATE, END_DATE)
    logger.info("Data loaded successfully.")
    data.save_dataset()
    logger.info("Data saved successfully.")

    # 2. train and run a model
    logger.info("Starting data transformation...")
    model = Model(data)
    model.define_categories()
    model.get_dummies()
    model.split_data()
    model.train_model()
    model.make_predictions()
    rf_model_pred = model.save_data_for_simulation()

    # 3. run a simulation
    simulation = StrategySimulation()
    logger.info("Starting simulation...")
    history, trades, metrics = simulation.backtest_fixed_capital(rf_model_pred)
    logger.info("Simulation complete. Metrics: %s", metrics)


if __name__ == "__main__":
    main()