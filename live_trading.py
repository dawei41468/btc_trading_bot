import ccxt
import schedule
import time
from config import BYBIT_API_KEY, BYBIT_API_SECRET, SYMBOL
from indicators import calculate_indicators
import pandas as pd
import xgboost as xgb

class LiveTrader:
    def __init__(self, data_handler, strategy, risk_manager, monitoring):
        """Initialize live trading with exchange and components."""
        self.exchange = ccxt.bybit({
            'apiKey': BYBIT_API_KEY,
            'secret': BYBIT_API_SECRET,
            'enableRateLimit': True
        })
        self.data_handler = data_handler
        self.strategy = strategy
        self.risk_manager = risk_manager
        self.monitoring = monitoring
        self.position = 0
        self.active_trade = None
        self.peak_price = 0
        self.cooldown = 0

    def sync_position(self):
        """Sync position and balance with the exchange."""
        try:
            balance = self.exchange.fetch_balance()
            self.position = balance['BTC']['free']
            self.risk_manager.initial_capital = balance['USDT']['free']
            self.monitoring.log(f"Synced position: {self.position} BTC, Balance: {self.risk_manager.initial_capital} USDT")
        except Exception as e:
            self.monitoring.log(f"Error syncing position: {e}")

    def trade(self):
        """Execute trades based on latest signals with ATR-based exits."""
        try:
            # Sync position before trading
            self.sync_position()

            # Fetch and prepare live data
            df = self.data_handler.fetch_live_data('4h', limit=200)
            if df.empty:
                self.monitoring.log("Failed to fetch live data")
                return
            df = calculate_indicators(df)
            X = df[ML_FEATURES]
            df['pred_prob'] = self.strategy.ml_model.predict(xgb.DMatrix(X))
            df['signal'] = (df['pred_prob'] > 0.65).astype(int)
            latest = df.iloc[-1]
            price = latest['close']
            atr = latest['ATR']
            ema20 = latest['EMA20']
            ema50 = latest['EMA50']
            ema200 = latest['EMA200']
            portfolio_value = self.risk_manager.initial_capital + (self.position * price)

            # Decrement cooldown
            if self.cooldown > 0:
                self.cooldown -= 1
                return

            # Check exits if holding a position
            if self.position > 0 and self.active_trade:
                if price < self.active_trade['price'] - atr:
                    self.exchange.create_market_sell_order(SYMBOL, self.position)
                    self.monitoring.log(f"Sell {self.position} BTC at {price} (stop-loss)")
                    self.position = 0
                    self.active_trade = None
                    self.peak_price = 0
                    self.cooldown = 2
                elif price > self.active_trade['price'] + atr_take_profit_multiplier * atr:
                    self.exchange.create_market_sell_order(SYMBOL, self.position)
                    self.monitoring.log(f"Sell {self.position} BTC at {price} (take-profit)")
                    self.position = 0
                    self.active_trade = None
                    self.peak_price = 0
                    self.cooldown = 2
                elif price < self.peak_price * (1 - TRAILING_STOP_PERCENT):
                    self.exchange.create_market_sell_order(SYMBOL, self.position)
                    self.monitoring.log(f"Sell {self.position} BTC at {price} (trailing-stop)")
                    self.position = 0
                    self.active_trade = None
                    self.peak_price = 0
                    self.cooldown = 2
                else:
                    self.peak_price = max(self.peak_price, price)

            # Enter new position if signal and sufficient funds
            if latest['signal'] == 1 and self.risk_manager.initial_capital >= POSITION_SIZE_FRACTION * portfolio_value and self.position == 0 and self.cooldown == 0:
                trade_value = POSITION_SIZE_FRACTION * portfolio_value
                amount = (trade_value / price) * (1 - TRANSACTION_FEE_RATE)
                self.exchange.create_market_buy_order(SYMBOL, amount)
                self.position = amount
                self.active_trade = {'price': price}
                self.peak_price = price
                self.monitoring.log(f"Buy {amount} BTC at {price}")

        except Exception as e:
            self.monitoring.log(f"Error in trade execution: {e}")
            time.sleep(60)  # Wait before retrying

    def run(self):
        """Run live trading on a 4-hour schedule."""
        schedule.every(4).hours.do(self.trade)
        while True:
            schedule.run_pending()
            time.sleep(1)