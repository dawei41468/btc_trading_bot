import ccxt
import pandas as pd
from config import BYBIT_API_KEY, BYBIT_API_SECRET, SYMBOL
import time
import os

class DataHandler:
    def __init__(self):
        self.exchange = ccxt.bybit({
            'apiKey': BYBIT_API_KEY,
            'secret': BYBIT_API_SECRET,
            'enableRateLimit': True
        })

    def fetch_historical_data(self, timeframe, start_date='2023-03-11 00:00:00', limit_per_call=1000):
        """Fetch historical OHLCV data for a given timeframe and save to CSV."""
        print(f"Fetching {timeframe} historical data from {start_date}...")
        start_timestamp = int(pd.to_datetime(start_date).timestamp() * 1000)
        end_timestamp = int(time.time() * 1000)
        all_ohlcv = []
        since = start_timestamp

        while since < end_timestamp:
            try:
                ohlcv = self.exchange.fetch_ohlcv(SYMBOL, timeframe, since=since, limit=limit_per_call)
                if not ohlcv:
                    break
                all_ohlcv.extend(ohlcv)
                since = ohlcv[-1][0] + 1
                print(f"Fetched {len(ohlcv)} candles up to {pd.to_datetime(since, unit='ms')}")
                time.sleep(1)
            except Exception as e:
                print(f"Error fetching {timeframe} data: {e}")
                break

        if not all_ohlcv:
            print(f"No {timeframe} data fetched.")
            return pd.DataFrame()

        df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        print(f"Total {timeframe} data points fetched: {len(df)}")
        print("First few rows:")
        print(df.head())

        filename = f"data_{SYMBOL.replace('/', '_')}_{timeframe}.csv"
        df.to_csv(filename, index=False)
        print(f"Saved {timeframe} data to {filename}")
        return df

    def load_historical_data(self, timeframe, period_name=None):
        """Load historical data from CSV if available, otherwise fetch it."""
        if period_name:
            filename = f"data_{SYMBOL.replace('/', '_')}_{timeframe}_{period_name}.csv"
        else:
            filename = f"data_{SYMBOL.replace('/', '_')}_{timeframe}.csv"
        if os.path.exists(filename):
            df = pd.read_csv(filename)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            print(f"Loaded {timeframe} data from {filename}. Total points: {len(df)}")
            print("First few rows:")
            print(df.head())
            return df
        else:
            print(f"No {timeframe} data file found for {filename}. Fetching data...")
            return self.fetch_historical_data(timeframe)

    def load_stress_data(self, timeframe, stress_type):
        """Load stress test data from CSV."""
        filename = f"data_{SYMBOL.replace('/', '_')}_{timeframe}_stress_{stress_type}.csv"
        if os.path.exists(filename):
            df = pd.read_csv(filename)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            print(f"Loaded {timeframe} stress data ({stress_type}) from {filename}. Total points: {len(df)}")
            print("First few rows:")
            print(df.head())
            return df
        else:
            print(f"No {timeframe} stress data file found for {stress_type}.")
            return pd.DataFrame()

    def fetch_live_data(self, timeframe, limit=200):
        """Fetch recent data for live trading (not saved to file)."""
        print(f"Fetching {timeframe} live data (last {limit} candles)...")
        ohlcv = self.exchange.fetch_ohlcv(SYMBOL, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df