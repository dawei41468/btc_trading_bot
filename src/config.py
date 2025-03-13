import os
from dotenv import load_dotenv

# Load environment variables from .env file in the project root
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'))

# Bybit API credentials (loaded from environment variables for security)
BYBIT_API_KEY = os.getenv('BYBIT_API_KEY')  # API key for Bybit exchange access
BYBIT_API_SECRET = os.getenv('BYBIT_API_SECRET')  # API secret for Bybit authentication

# Trading parameters
SYMBOL = 'BTC/USDT'  # Trading pair for Bitcoin vs. Tether
TIMEFRAME = '4h'  # Timeframe for data and trading signals (4-hour candles)
INITIAL_CAPITAL = 10000  # Starting capital in USDT
POSITION_SIZE_FRACTION = 0.70  # Position size as 70% of current portfolio value (replaces POSITION_SIZE_PERCENT)
TRANSACTION_FEE_RATE = 0.000775  # Bybit fee rate (0.0775% per trade)

# Indicator parameters for technical analysis
SMA_PERIOD = 50  # Period for Simple Moving Average
RSI_PERIOD = 20  # Period for Relative Strength Index, tuned for 4h timeframe
MACD_FAST = 12  # Fast EMA period for MACD
MACD_SLOW = 26  # Slow EMA period for MACD
MACD_SIGNAL = 9  # Signal line period for MACD
BB_PERIOD = 20  # Period for Bollinger Bands
BB_STD = 2  # Standard deviations for Bollinger Bands width

# ML Parameters for XGBoost model
ML_FEATURES = [
    'RSI', 'MACD_signal', 'BB_width', 'volume_change', 'close_change',
    'bb_position', 'return_lag1', 'ATR', 'ATR_normalized', 'ema_diff',
    'momentum', 'momentum_vol_adj', 'EMA100', 'SMA200', 'volatility_change',
    'volume_momentum', 'price_volume_corr', 'MACD_histogram_slope'
    # Matches 101.93%, 77.55%, 66.48%, and 55.40% feature set
]
ML_TEST_SIZE = 0.15  # Fraction of data for test set (15%)
ML_N_ESTIMATORS = 500  # Number of boosting rounds (unused in xgb.train, kept for reference)
ML_MAX_DEPTH = 6  # Maximum tree depth for XGBoost
ML_LEARNING_RATE = 0.01  # Learning rate (unused in xgb.train, kept for reference)
EARLY_STOPPING_ROUNDS = 50  # Rounds for early stopping in training

# Risk management parameters
STOP_LOSS_PERCENT = 0.05  # Fixed stop-loss at 5% (unused with ATR-based stop)
DAILY_LOSS_LIMIT = 0.05  # Daily loss limit at 5% (not implemented in backtest)
TRAILING_STOP_PERCENT = 0.03  # Trailing stop at 3% from peak
TAKE_PROFIT_PERCENT = 0.05  # Fixed take-profit at 5% (unused with ATR-based profit)