from data_handler import DataHandler
from indicators import calculate_indicators
from strategy import TradingStrategy
from ml_model import MLModel
from backtest import backtest
from risk_management import RiskManager
from live_trading import LiveTrader
from monitoring import Monitoring
import argparse
import pandas as pd

last_backtest_results = {}

def filter_portfolio_changes(portfolio_values):
    if not portfolio_values:
        return []
    filtered = [portfolio_values[0]]
    for i in range(1, len(portfolio_values)):
        if portfolio_values[i] != portfolio_values[i - 1]:
            filtered.append(portfolio_values[i])
    return filtered

def main(mode):
    global last_backtest_results
    data_handler = DataHandler()
    ml_model = MLModel()
    strategy = TradingStrategy(ml_model)
    risk_manager = RiskManager(10000)
    monitoring = Monitoring()

    if mode == 'train':
        df = data_handler.fetch_historical_data(start_date='2023-03-11 00:00:00')
        print(f"Training Data Shape: {df.shape}")
        print(f"First few rows:\n{df.head()}")
        df = calculate_indicators(df)
        print(f"Indicators Calculated. Shape: {df.shape}")
        print(f"Sample indicators:\n{df[['close', 'SMA50', 'RSI', 'MACD']].tail()}")
        accuracy = ml_model.train(df)
        print(f"ML Model Trained. Accuracy: {accuracy:.2f}")

    elif mode == 'backtest':
        df_full = data_handler.fetch_historical_data(start_date='2023-03-11 00:00:00')
        print(f"Full Data Shape: {df_full.shape}")
        print(f"First few rows:\n{df_full.head()}")
        if df_full.empty:
            print("Error: No data fetched for backtest.")
            return

        df_full = calculate_indicators(df_full)
        print(f"Indicators Calculated (Full). Shape: {df_full.shape}")
        accuracy = ml_model.train(df_full)
        print(f"ML Model Trained on Full Data. Accuracy: {accuracy:.2f}")

        backtest_start = pd.to_datetime('2024-03-11 00:00:00')
        df = df_full[df_full['timestamp'] >= backtest_start].copy()
        print(f"Backtest Data Shape (1 year): {df.shape}")
        print(f"First few rows of backtest data:\n{df.head()}")

        df = calculate_indicators(df)
        print(f"Indicators Calculated (Backtest). Shape: {df.shape}")
        print(f"Sample indicators:\n{df[['close', 'SMA50', 'RSI', 'MACD']].tail()}")
        df = strategy.generate_signals(df)
        print(f"Signals Generated. Sample signals:\n{df[['close', 'buy_signal', 'sell_signal', 'ml_pred', 'final_buy_signal', 'final_sell_signal']].tail()}")
        results = backtest(df)

        changed_results = {}
        print("Backtest Results:")
        for key, value in results.items():
            if key == 'portfolio_values':
                filtered_value = filter_portfolio_changes(value)
                if key not in last_backtest_results or filter_portfolio_changes(last_backtest_results.get(key, [])) != filtered_value:
                    changed_results[key] = filtered_value
            elif key == 'trades':
                if key not in last_backtest_results or len(last_backtest_results[key]) != len(value) or last_backtest_results[key] != value:
                    changed_results[key] = value
            else:
                if key not in last_backtest_results or last_backtest_results[key] != value:
                    changed_results[key] = value

        if changed_results:
            for key, value in changed_results.items():
                print(f"  {key}: {value}")
        else:
            print("  No changes from previous backtest results.")
        last_backtest_results = results.copy()

    elif mode == 'live':
        df = data_handler.fetch_historical_data()
        df = calculate_indicators(df)
        accuracy = ml_model.train(df)
        print(f"ML Model Trained. Accuracy: {accuracy:.2f}")
        trader = LiveTrader(data_handler, strategy, risk_manager, monitoring)
        trader.run()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BTC Trading Bot')
    parser.add_argument('mode', choices=['train', 'backtest', 'live'], help='Mode to run')
    args = parser.parse_args()
    main(args.mode)