import ta
import pandas as pd
from config import SMA_PERIOD, RSI_PERIOD, MACD_FAST, MACD_SLOW, MACD_SIGNAL, BB_PERIOD, BB_STD

def calculate_indicators(df):
    """Calculate technical indicators for ML features and trading signals."""
    # Moving Averages for trend analysis
    df['SMA50'] = ta.trend.sma_indicator(df['close'], window=SMA_PERIOD)
    df['EMA20'] = ta.trend.ema_indicator(df['close'], window=20)
    df['EMA50'] = ta.trend.ema_indicator(df['close'], window=50)
    df['EMA100'] = ta.trend.ema_indicator(df['close'], window=100)
    df['SMA200'] = ta.trend.sma_indicator(df['close'], window=200)
    df['EMA200'] = ta.trend.ema_indicator(df['close'], window=200)  # For long-term trend detection

    # Momentum and oscillator indicators
    df['RSI'] = ta.momentum.rsi(df['close'], window=RSI_PERIOD)

    # MACD for trend and momentum
    macd = ta.trend.MACD(df['close'], window_fast=MACD_FAST, window_slow=MACD_SLOW, window_sign=MACD_SIGNAL)
    df['MACD'] = macd.macd()
    df['MACD_signal'] = macd.macd_signal()
    df['MACD_diff'] = macd.macd_diff()
    df['MACD_histogram_slope'] = df['MACD_diff'].diff()

    # Bollinger Bands for volatility and price position
    df['BB_upper'] = ta.volatility.bollinger_hband(df['close'], window=BB_PERIOD, window_dev=BB_STD)
    df['BB_lower'] = ta.volatility.bollinger_lband(df['close'], window=BB_PERIOD, window_dev=BB_STD)
    df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['close']
    df['bb_position'] = (df['close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])

    # Volume and price change features
    df['volume_change'] = df['volume'].pct_change()
    df['volume_momentum'] = ta.momentum.roc(df['volume'], window=20)
    df['close_change'] = df['close'].pct_change()
    df['return_lag1'] = df['close'].pct_change().shift(1)

    # Volatility measures
    df['ATR'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)
    df['ATR_normalized'] = df['ATR'] / df['close']
    df['volatility_change'] = df['ATR_normalized'].pct_change()

    # Price-volume correlation
    df['price_volume_corr'] = df['close'].rolling(window=20).corr(df['volume'])

    # EMA difference for trend signal
    df['ema_diff'] = df['EMA20'] - df['EMA50']

    # Momentum indicators
    df['momentum'] = ta.momentum.roc(df['close'], window=20)
    df['momentum_vol_adj'] = df['momentum'] / df['ATR_normalized']

    # Clean up NaN values from calculations
    df = df.dropna()
    print(f"Indicators Calculated. Shape: {df.shape}")
    print("Sample indicators:")
    print(df[['close', 'SMA50', 'RSI', 'MACD', 'EMA100', 'SMA200', 'EMA200']].tail())
    return df