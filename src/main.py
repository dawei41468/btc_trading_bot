import sys
from src.data_handler import DataHandler
from src.indicators import calculate_indicators
from src.ml_model import MLModel
from src.backtest_utils import backtest
from src.config import TIMEFRAME

def fetch_and_save_all_timeframes():
    """Fetch and save historical data for multiple timeframes."""
    handler = DataHandler()
    timeframes = ['4h', '1h', '15m']
    for tf in timeframes:
        handler.fetch_historical_data(tf)

def train(timeframe=TIMEFRAME):
    """Train the ML model on historical data."""
    handler = DataHandler()
    df = handler.load_historical_data(timeframe)
    if df.empty:
        print(f"No {timeframe} data available for training.")
        return
    df = calculate_indicators(df)
    model = MLModel()
    accuracy = model.train(df)
    print(f"ML Model Trained on {timeframe} data. Accuracy: {accuracy:.2f}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.main [fetch|train|backtest] [optional: timeframe]")
        sys.exit(1)
    command = sys.argv[1].lower()
    if command == "fetch":
        fetch_and_save_all_timeframes()
    elif command == "train":
        timeframe = sys.argv[2] if len(sys.argv) > 2 else TIMEFRAME
        train(timeframe)
    elif command == "backtest":
        timeframe = sys.argv[2] if len(sys.argv) > 2 else TIMEFRAME
        backtest(timeframe)
    else:
        print(f"Unknown command: {command}. Use 'fetch', 'train', or 'backtest'.")