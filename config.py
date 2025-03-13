import os
from dotenv import load_dotenv

# Load environment variables from .env file for secure API key storage
load_dotenv()

# Bybit API credentials (loaded from environment variables for security)
BYBIT_API_KEY = os.getenv('BYBIT_API_KEY')
BYBIT_API_SECRET = os.getenv('BYBIT_API_SECRET')

# Trading parameters
SYMBOL = 'BTC/USDT'
TIMEFRAME = '4h'
INITIAL_CAPITAL = 10000
POSITION_SIZE_FRACTION = 0.75
TRANSACTION_FEE_RATE = 0.000775

# Indicator parameters for technical analysis
SMA_PERIOD = 50
RSI_PERIOD = 20
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
BB_PERIOD = 20
BB_STD = 2

# ML Parameters for XGBoost model
ML_FEATURES = [
    'RSI', 'MACD_signal', 'BB_width', 'volume_change', 'close_change',
    'bb_position', 'return_lag1', 'ATR', 'ATR_normalized', 'ema_diff',
    'momentum', 'momentum_vol_adj', 'EMA100', 'SMA200', 'volatility_change',
    'volume_momentum', 'price_volume_corr', 'MACD_histogram_slope'
]
ML_TEST_SIZE = 0.15
ML_N_ESTIMATORS = 500
ML_MAX_DEPTH = 6
ML_LEARNING_RATE = 0.01
EARLY_STOPPING_ROUNDS = 50

# Risk management parameters
STOP_LOSS_PERCENT = 0.05
DAILY_LOSS_LIMIT = 0.05
TRAILING_STOP_PERCENT = 0.03
TAKE_PROFIT_PERCENT = 0.05

# Dynamic ATR Take-Profit Multipliers for Hybrid Trend Detection
ATR_TAKE_PROFIT_UPTREND = 5.0    # Long-term uptrend (EMA50 > EMA200)
ATR_TAKE_PROFIT_NON_TREND = 3.0  # Short-term bear/sideways (EMA20 <= EMA50)
ATR_TAKE_PROFIT_INTERMEDIATE = 4.0  # Short-term uptrend within non-uptrend