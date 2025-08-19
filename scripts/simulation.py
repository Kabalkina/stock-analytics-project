# scripts/3_simulation.py
import pandas as pd
import numpy as np
import os
import json

from config_loader import INITIAL_CAPITAL, TRANSACTION_FEE, STOP_LOSS_PCT, TAKE_PROFIT_PCT


import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("simulation_pipeline")


stop_losses=[-0.20, -0.10, -0.05]
take_profits=[0.10, 0.20, 0.30]

class StrategySimulation:

    initial_capital: int = INITIAL_CAPITAL
    transaction_fee: float = TRANSACTION_FEE
    stop_loss_pct: float = STOP_LOSS_PCT
    take_profit_pct: float = TAKE_PROFIT_PCT

    portfolio_history: pd.DataFrame
    trades: pd.DataFrame
    metrics: pd.DataFrame

    results_df: pd.DataFrame
    best_row: pd.Series

    def __init__(self):

        self.portfolio_history = None
        self.trades = None
        self.metrics = None
        self.results_df = None
        self.best_row = None

    # -----------------------------------------------------------------------------
    # Run simulation
    # -----------------------------------------------------------------------------
    def backtest_fixed_capital(
            self,
            df,
            initial_capital=INITIAL_CAPITAL,
            transaction_fee=TRANSACTION_FEE,
            stop_loss_pct=STOP_LOSS_PCT,
            take_profit_pct=TAKE_PROFIT_PCT
        ):
        """
        Run trading backtest with fixed capital allocation.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain columns: 'date', 'ticker', 'close', 'pred_rf'
        initial_capital : float
            Starting capital.
        transaction_fee : float
            Fee per transaction.
        stop_loss_pct : float
            Stop loss threshold.
        take_profit_pct : float
            Take profit threshold.

        Returns
        -------
        portfolio_history : pd.DataFrame
        trades : pd.DataFrame
        metrics : dict
        """
        df = df.sort_values(["date", "ticker"]).reset_index(drop=True)
        cash = initial_capital
        positions = {}
        portfolio_values, trade_log = [], []

        logger.info(f"Running backtest: stop_loss={stop_loss_pct}, take_profit={take_profit_pct}, fee={transaction_fee}")

        for date, group in df.groupby("date"):
            # --- check stop loss/take profit ---
            tickers_to_sell = []
            for ticker, pos in positions.items():
                current_price = group.loc[group["ticker"] == ticker, "close"].values
                if len(current_price) == 0:
                    continue
                current_price = current_price[0]
                change_pct = (current_price - pos["buy_price"]) / pos["buy_price"]

                if change_pct <= stop_loss_pct or change_pct >= take_profit_pct:
                    sell_value = pos["shares"] * current_price
                    cash += sell_value - transaction_fee
                    trade_log.append({"date": date, "ticker": ticker, "type": "SELL", "pnl": change_pct})
                    tickers_to_sell.append(ticker)

            for ticker in tickers_to_sell:
                positions.pop(ticker)

            # --- buy signals ---
            buy_signals = group[group["pred_rf"] == True]["ticker"].tolist()
            buy_signals = [t for t in buy_signals if t not in positions]

            if buy_signals and cash > 0:
                allocation = cash / len(buy_signals)
                for ticker in buy_signals:
                    price = group.loc[group["ticker"] == ticker, "close"].values[0]
                    shares = (allocation - transaction_fee) / price
                    if shares <= 0:
                        continue
                    positions[ticker] = {"buy_price": price, "shares": shares, "buy_date": date}
                    trade_log.append({"date": date, "ticker": ticker, "type": "BUY", "pnl": 0})
                    cash -= allocation

            # --- portfolio value ---
            value_positions = 0
            for ticker, pos in positions.items():
                price = group.loc[group["ticker"] == ticker, "close"].values
                if len(price) == 0:
                    continue
                value_positions += pos["shares"] * price[0]

            portfolio_value = cash + value_positions
            portfolio_values.append({"date": date, "portfolio_value": portfolio_value, "cash": cash})

        portfolio_history = pd.DataFrame(portfolio_values)
        trades = pd.DataFrame(trade_log)

        portfolio_history["date"] = pd.to_datetime(portfolio_history["date"])
        final_value = portfolio_history["portfolio_value"].iloc[-1]
        years = (portfolio_history["date"].iloc[-1] - portfolio_history["date"].iloc[0]).days / 365.25
        cagr = (final_value / initial_capital) ** (1 / years) - 1

        metrics = {
            "final_value": final_value,
            "cagr": cagr * 100,
            "num_trades": len(trades)
        }
        
        self.portfolio_history = portfolio_history
        self.trades = trades
        self.metrics = metrics

        logger.info(f"Backtest complete: final value={final_value:.2f}, CAGR={cagr:.2f}%, trades={len(trades)}")
        
        return self.portfolio_history, self.trades, self.metrics

    # -----------------------------------------------------------------------------
    # Tune simulation parameters
    # -----------------------------------------------------------------------------

    def tune_params(self,
                    df, 
                    initial_capital=INITIAL_CAPITAL, 
                    transaction_fee=TRANSACTION_FEE, 
                    stop_losses=stop_losses, 
                    take_profits=take_profits):
        """
        Runs a grid search over stop losses and take profit values
        to find the strategy with the highest CAGR.
        """
        results = []

        logger.info("Starting parameter tuning")

        for sl in stop_losses:
            for tp in take_profits:
                _, _, metrics = self.backtest_fixed_capital(
                    df,
                    initial_capital=initial_capital,
                    transaction_fee=transaction_fee,
                    stop_loss_pct=sl,
                    take_profit_pct=tp
                )
                results.append({
                    'stop_loss': sl,
                    'take_profit': tp,
                    **metrics
                })

        results_df = pd.DataFrame(results)

        # Find best CAGR
        best_row = results_df.loc[results_df['cagr'].idxmax()]
        
        self.results_df = results_df
        self.best_row = best_row

        logger.info(f"Best params: stop_loss={best_row['stop_loss']}, take_profit={best_row['take_profit']}, CAGR={best_row['cagr']:.2f}%")


    # -----------------------------------------------------------------------------
    # Save dataset
    # -----------------------------------------------------------------------------
    def save_dataset(self):
        """Save dataframes to files in the local 'Data' directory"""
        # Resolve path relative to this script's location
        script_dir = os.path.dirname(os.path.abspath(__file__))  # .../scripts
        data_dir = os.path.join(script_dir, "../Data")
        os.makedirs(data_dir, exist_ok=True)

        # Only save if dataframes exist and are not empty
        if self.portfolio_history is not None and not self.portfolio_history.empty:
            file_name_portfolio_history = "portfolio_history.csv"
            file_name_trades = "trades.csv"
            file_name_metrics = "metrics.json"
            #file_path = os.path.join(data_dir, file_name)

            if os.path.exists(os.path.join(data_dir, file_name_portfolio_history)):
                os.remove(os.path.join(data_dir, file_name_portfolio_history))

            self.portfolio_history.to_csv(os.path.join(data_dir, file_name_portfolio_history), index=False)
            self.trades.to_csv(os.path.join(data_dir, file_name_trades), index=False)
            with open(os.path.join(data_dir, file_name_metrics), "w") as f:
                json.dump(self.metrics, f)
            logger.info(f"Saved simulation results to {data_dir}")
        else:
            logger.info("No simulation results to save")

