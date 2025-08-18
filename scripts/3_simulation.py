# scripts/3_simulation.py
import pandas as pd
import numpy as np


def backtest_fixed_capital(
    df,
    initial_capital=1000,
    transaction_fee=2.0,
    stop_loss_pct=-0.20,
    take_profit_pct=0.30
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
    return portfolio_history, trades, metrics


if __name__ == "__main__":
    df = pd.read_csv("data/simulation_input.csv")
    history, trades, metrics = backtest_fixed_capital(df)
    print("Simulation complete. Metrics:", metrics)
