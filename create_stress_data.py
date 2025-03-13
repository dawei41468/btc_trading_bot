import pandas as pd

# Load base sideways data
df = pd.read_csv('data_BTC_USDT_4h_sideways.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Insert flash crash at midpoint
midpoint = len(df) // 2
crash_row = df.iloc[midpoint - 1].copy()
crash_row['open'] = crash_row['close']
crash_row['high'] = crash_row['close']
crash_row['low'] = crash_row['close'] * 0.8  # 20% drop
crash_row['close'] = crash_row['low']
crash_row['volume'] *= 2
crash_row['timestamp'] = crash_row['timestamp'] + pd.Timedelta(hours=4)

recovery_row = crash_row.copy()
recovery_row['open'] = crash_row['close']
recovery_row['high'] = crash_row['open'] * 1.15
recovery_row['low'] = crash_row['open']
recovery_row['close'] = crash_row['open'] * 1.10
recovery_row['volume'] *= 1.5
recovery_row['timestamp'] = crash_row['timestamp'] + pd.Timedelta(hours=4)

df = pd.concat([df.iloc[:midpoint], pd.DataFrame([crash_row, recovery_row]), df.iloc[midpoint:]]).reset_index(drop=True)
df.to_csv('data_BTC_USDT_4h_stress_flash_crash.csv', index=False)
print("Flash crash stress data created: data_BTC_USDT_4h_stress_flash_crash.csv")