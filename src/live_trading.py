import ccxt
import pandas as pd
import time
import os
from dotenv import load_dotenv
from src.config import SYMBOL, TIMEFRAME, INITIAL_CAPITAL, POSITION_SIZE_FRACTION, TRANSACTION_FEE_RATE, TRAILING_STOP_PERCENT, ML_FEATURES
from src.data_handler import DataHandler
from src.indicators import calculate_indicators
from src.ml_model import MLModel
from src.trade_utils import execute_buy_trade, execute_sell_trade
import xgboost as xgb

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

class LiveTrader:
    def __init__(self):
        # Initialize Bybit testnet exchange
        self.exchange = ccxt.bybit({
            'apiKey': os.environ.get('BYBIT_API_KEY'),
            'secret': os.environ.get('BYBIT_API_SECRET'),
            'enableRateLimit': True,
            'test': True  # Use testnet
        })
        self.data_handler = DataHandler()
        self.model = MLModel()
        self.cash = INITIAL_CAPITAL  # Starting USDT (testnet funds)
        self.position = 0  # BTC held
        self.active_trade = None
        self.peak_price = 0
        self.cooldown = 0
        self.trade_number = 0
        # Metrics for tracking trades (simplified from backtest)
        self.trend_metrics = {
            'gross_profit': 0, 'gross_loss': 0, 'consecutive_wins': 0, 'consecutive_losses': 0,
            'max_consecutive_wins': 0, 'max_consecutive_losses': 0, 'holding_periods': [],
            'trend_regime_profits': {'trending': 0, 'choppy': 0},
            'trend_regime_trades': {'trending': 0, 'choppy': 0},
            'trend_regime_wins': {'trending': 0, 'choppy': 0},
            'regime': None
        }

    def sync_position(self):
        """Sync position and balance with the exchange."""
        try:
            balance = self.exchange.fetch_balance()
            self.position = balance.get('BTC', {}).get('free', 0)
            self.cash = balance.get('USDT', {}).get('free', INITIAL_CAPITAL)
            print(f"Synced position: {self.position:.6f} BTC, Cash: {self.cash:.2f} USDT")
        except Exception as e:
            print(f"Error syncing position: {e}")
            self.position = 0
            self.cash = INITIAL_CAPITAL

    def fetch_latest_data(self):
        """Fetch the latest 200 candles for analysis."""
        df = self.data_handler.fetch_live_data(TIMEFRAME, limit=200)
        if df.empty:
            print("No live data fetched.")
            return pd.DataFrame()
        df = calculate_indicators(df)
        return df

    def run(self):
        """Main trading loop."""
        print(f"Starting paper trading on {SYMBOL} ({TIMEFRAME}) with Bybit testnet...")
        while True:
            try:
                # Sync position and balance
                self.sync_position()

                # Fetch and prepare latest data
                df = self.fetch_latest_data()
                if df.empty:
                    print("Skipping cycle: No data available.")
                    time.sleep(60)
                    continue

                # ML prediction
                X = df[ML_FEATURES].iloc[-1:].values
                df['pred_prob'] = self.model.predict(xgb.DMatrix(X))
                df['signal'] = (df['pred_prob'] > 0.65).astype(int)
                latest = df.iloc[-1]
                price = latest['close']
                atr = latest['ATR']
                regime = 'trending' if latest['SMA50'] > latest['SMA200'] else 'choppy'
                self.trend_metrics['regime'] = regime
                signal = latest['signal']
                portfolio_value = self.cash + self.position * price

                # Log current state
                print(f"Timestamp: {latest['timestamp']}, Price: {price:.2f}, Signal: {signal}, "
                      f"Position: {self.position:.6f}, Cash: {self.cash:.2f}, Regime: {regime}")

                # Handle cooldown
                if self.cooldown > 0:
                    self.cooldown -= 1
                    print(f"Cooldown: {self.cooldown} candles remaining")
                    time.sleep(60 * 60 * 4)  # Wait for 4 hours
                    continue

                # Check for exits if holding a position
                if self.position > 0 and self.active_trade:
                    stop_loss_multiplier = 0.5 if regime == 'choppy' else 1.0
                    if price < self.active_trade['price'] - atr * stop_loss_multiplier:
                        # Execute sell via exchange
                        order = self.exchange.create_market_sell_order(SYMBOL, self.position)
                        trade, cash_gain, cooldown = execute_sell_trade(
                            self.trade_number + 1, self.active_trade, price, self.position,
                            latest['timestamp'], portfolio_value, 'stop-loss', self.trend_metrics
                        )
                        print(f"Sell order executed: {order}")
                        self.cash += cash_gain
                        self.position = 0
                        self.active_trade = None
                        self.peak_price = 0
                        self.cooldown = cooldown
                        self.trade_number += 1
                    elif price > self.active_trade['price'] + 5 * atr:
                        order = self.exchange.create_market_sell_order(SYMBOL, self.position)
                        trade, cash_gain, cooldown = execute_sell_trade(
                            self.trade_number + 1, self.active_trade, price, self.position,
                            latest['timestamp'], portfolio_value, 'take-profit', self.trend_metrics
                        )
                        print(f"Sell order executed: {order}")
                        self.cash += cash_gain
                        self.position = 0
                        self.active_trade = None
                        self.peak_price = 0
                        self.cooldown = cooldown
                        self.trade_number += 1
                    elif price < self.peak_price * (1 - TRAILING_STOP_PERCENT):
                        order = self.exchange.create_market_sell_order(SYMBOL, self.position)
                        trade, cash_gain, cooldown = execute_sell_trade(
                            self.trade_number + 1, self.active_trade, price, self.position,
                            latest['timestamp'], portfolio_value, 'trailing-stop', self.trend_metrics
                        )
                        print(f"Sell order executed: {order}")
                        self.cash += cash_gain
                        self.position = 0
                        self.active_trade = None
                        self.peak_price = 0
                        self.cooldown = cooldown
                        self.trade_number += 1
                    else:
                        self.peak_price = max(self.peak_price, price)
                elif signal == 1 and self.position == 0:
                    # Execute buy via exchange
                    trade_info, amount, new_cash = execute_buy_trade(
                        self.trade_number + 1, signal, price, portfolio_value, self.cash, latest['timestamp'], regime
                    )
                    if trade_info:
                        order = self.exchange.create_market_buy_order(SYMBOL, amount)
                        print(f"Buy order executed: {order}")
                        self.position = amount
                        self.cash = new_cash
                        self.active_trade = trade_info
                        self.peak_price = price
                        self.trade_number += 1

                time.sleep(60 * 60 * 4)  # Wait 4 hours (adjustable)

            except Exception as e:
                print(f"Error in trading loop: {e}")
                time.sleep(60)  # Retry after 1 minute

if __name__ == "__main__":
    trader = LiveTrader()
    trader.run()