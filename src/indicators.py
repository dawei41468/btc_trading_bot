import ta
import pandas as pd
from src.config import SMA_PERIOD, RSI_PERIOD, MACD_FAST, MACD_SLOW, MACD_SIGNAL, BB_PERIOD, BB_STD

def calculate_indicators(df):
    """Calculate technical indicators for ML features and trading signals."""
    df['SMA50'] = ta.trend.sma_indicator(df['close'], window=SMA_PERIOD)
    df['EMA20'] = ta.trend.ema_indicator(df['close'], window=20)
    df['EMA50'] = ta.trend.ema_indicator(df['close'], window=50)
    df['EMA100'] = ta.trend.ema_indicator(df['close'], window=100)
    df['SMA200'] = ta.trend.sma_indicator(df['close'], window=200)
    df['RSI'] = ta.momentum.rsi(df['close'], window=RSI_PERIOD)
    macd = ta.trend.MACD(df['close'], window_fast=MACD_FAST, window_slow=MACD_SLOW, window_sign=MACD_SIGNAL)
    df['MACD'] = macd.macd()
    df['MACD_signal'] = macd.macd_signal()
    df['MACD_diff'] = macd.macd_diff()
    df['MACD_histogram_slope'] = df['MACD_diff'].diff()
    df['BB_upper'] = ta.volatility.bollinger_hband(df['close'], window=BB_PERIOD, window_dev=BB_STD)
    df['BB_lower'] = ta.volatility.bollinger_lband(df['close'], window=BB_PERIOD, window_dev=BB_STD)
    df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['close']
    df['bb_position'] = (df['close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])
    df['volume_change'] = df['volume'].pct_change()
    df['volume_momentum'] = ta.momentum.roc(df['volume'], window=20)
    df['close_change'] = df['close'].pct_change()
    df['return_lag1'] = df['close'].pct_change().shift(1)
    df['ATR'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)
    df['ATR_normalized'] = df['ATR'] / df['close']
    df['volatility_change'] = df['ATR_normalized'].pct_change()
    df['price_volume_corr'] = df['close'].rolling(window=20).corr(df['volume'])
    df['ema_diff'] = df['EMA20'] - df['EMA50']
    df['momentum'] = ta.momentum.roc(df['close'], window=20)
    df['momentum_vol_adj'] = df['momentum'] / df['ATR_normalized']
    df = df.dropna()
    print("Feature Distributions (after NaN removal):")
    key_features = ['RSI', 'ATR_normalized', 'MACD_diff', 'BB_width', 'volume_change']
    for feature in key_features:
        print(f"{feature}:")
        print(f"  Mean: {df[feature].mean():.4f}")
        print(f"  Std: {df[feature].std():.4f}")
        print(f"  Min: {df[feature].min():.4f}")
        print(f"  Max: {df[feature].max():.4f}")
        print(f"  Quantiles (25%, 50%, 75%): {df[feature].quantile([0.25, 0.50, 0.75]).to_dict()}")
    print(f"Indicators Calculated. Shape: {df.shape}")
    print("Sample indicators:")
    print(df[['close', 'SMA50', 'RSI', 'MACD', 'EMA100', 'SMA200']].tail())
    return df