import pandas as pd
import numpy as np

df = pd.read_csv('data_BTC_USDT_4h_sideways.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Simulate high volatility for 10 candles at midpoint
midpoint = len(df) // 2
for i in range(midpoint, midpoint + 10):
    if i < len(df):
        volatility_factor = np.random.uniform(1.5, 2.5)  # 1.5x to 2.5x ATR swing
        df.at[i, 'high'] = df.at[i, 'close'] * volatility_factor
        df.at[i, 'low'] = df.at[i, 'close'] / volatility_factor
        df.at[i, 'close'] = df.at[i, 'close'] * np.random.uniform(0.9, 1.1)  # Random close within range
        df.at[i, 'volume'] *= 2  # Increased volume

df.to_csv('data_BTC_USDT_4h_stress_high_volatility.csv', index=False)
print("High volatility stress data created: data_BTC_USDT_4h_stress_high_volatility.csv")