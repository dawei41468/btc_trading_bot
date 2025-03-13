import ccxt
import schedule
import time
from config import (
    BYBIT_API_KEY, BYBIT_API_SECRET, SYMBOL, ML_FEATURES,
    POSITION_SIZE_FRACTION, TRANSACTION_FEE_RATE, TRAILING_STOP_PERCENT,
    ATR_TAKE_PROFIT_UPTREND, ATR_TAKE_PROFIT_NON_TREND, ATR_TAKE_PROFIT_INTERMEDIATE
)
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
        # Added retry configuration for error handling
        self.retry_attempts = 3
        self.retry_delay = 5  # Seconds between retries

    def sync_position(self):
        """Sync position and balance with the exchange with retry logic."""
        for attempt in range(self.retry_attempts):
            try:
                balance = self.exchange.fetch_balance()
                # Safely access BTC and USDT balances, default to 0 if not found
                self.position = balance.get('BTC', {}).get('free', 0)
                self.risk_manager.initial_capital = balance.get('USDT', {}).get('free', 0)
                self.monitoring.log(f"Synced position: {self.position} BTC, Balance: {self.risk_manager.initial_capital} USDT")
                return True
            except Exception as e:
                self.monitoring.log(f"Error syncing position (attempt {attempt + 1}/{self.retry_attempts}): {str(e)}")
                if attempt < self.retry_attempts - 1:
                    time.sleep(self.retry_delay)
                else:
                    self.monitoring.log("Max retries reached for sync_position, setting defaults")
                    self.position = 0
                    self.risk_manager.initial_capital = 0
                    return False

    def trade(self):
        """Execute trades based on latest signals with ATR-based exits."""
        try:
            # Sync position before trading
            if not self.sync_position():
                self.monitoring.log("Failed to sync position, skipping trade execution")
                return

            # Fetch and prepare live data with retry logic
            for attempt in range(self.retry_attempts):
                try:
                    df = self.data_handler.fetch_live_data('4h', limit=200)
                    if df.empty:
                        self.monitoring.log("Fetched live data is empty")
                        return
                    break
                except Exception as e:
                    self.monitoring.log(f"Error fetching live data (attempt {attempt + 1}/{self.retry_attempts}): {str(e)}")
                    if attempt < self.retry_attempts - 1:
                        time.sleep(self.retry_delay)
                    else:
                        self.monitoring.log("Max retries reached for fetch_live_data, skipping trade")
                        return

            # Calculate indicators
            df = calculate_indicators(df)
            if df.empty:
                self.monitoring.log("Failed to calculate indicators, skipping trade")
                return

            # Prepare ML features and predictions
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

            # Set ATR take-profit multiplier based on trend
            if ema50 > ema200:  # Long-term uptrend
                atr_take_profit_multiplier = ATR_TAKE_PROFIT_UPTREND
            elif ema20 <= ema50:  # Short-term bear/sideways
                atr_take_profit_multiplier = ATR_TAKE_PROFIT_NON_TREND
            else:  # Short-term uptrend within long-term non-uptrend
                atr_take_profit_multiplier = ATR_TAKE_PROFIT_INTERMEDIATE

            # Decrement cooldown
            if self.cooldown > 0:
                self.cooldown -= 1
                self.monitoring.log(f"Cooldown active: {self.cooldown} periods remaining")
                return

            # Check exits if holding a position
            if self.position > 0 and self.active_trade:
                if price < self.active_trade['price'] - atr:  # 1x ATR stop-loss
                    for attempt in range(self.retry_attempts):
                        try:
                            order = self.exchange.create_market_sell_order(SYMBOL, self.position)
                            self.monitoring.log(f"Sell {self.position} BTC at {price} (stop-loss), order: {order}")
                            self.position = 0
                            self.active_trade = None
                            self.peak_price = 0
                            self.cooldown = 2
                            break
                        except Exception as e:
                            self.monitoring.log(f"Error executing sell order (stop-loss, attempt {attempt + 1}/{self.retry_attempts}): {str(e)}")
                            if attempt < self.retry_attempts - 1:
                                time.sleep(self.retry_delay)
                            else:
                                self.monitoring.log("Max retries reached for stop-loss sell order")
                                return
                elif price > self.active_trade['price'] + atr_take_profit_multiplier * atr:  # Hybrid ATR take-profit
                    for attempt in range(self.retry_attempts):
                        try:
                            order = self.exchange.create_market_sell_order(SYMBOL, self.position)
                            self.monitoring.log(f"Sell {self.position} BTC at {price} (take-profit), order: {order}")
                            self.position = 0
                            self.active_trade = None
                            self.peak_price = 0
                            self.cooldown = 2
                            break
                        except Exception as e:
                            self.monitoring.log(f"Error executing sell order (take-profit, attempt {attempt + 1}/{self.retry_attempts}): {str(e)}")
                            if attempt < self.retry_attempts - 1:
                                time.sleep(self.retry_delay)
                            else:
                                self.monitoring.log("Max retries reached for take-profit sell order")
                                return
                elif price < self.peak_price * (1 - TRAILING_STOP_PERCENT):  # Trailing stop
                    for attempt in range(self.retry_attempts):
                        try:
                            order = self.exchange.create_market_sell_order(SYMBOL, self.position)
                            self.monitoring.log(f"Sell {self.position} BTC at {price} (trailing-stop), order: {order}")
                            self.position = 0
                            self.active_trade = None
                            self.peak_price = 0
                            self.cooldown = 2
                            break
                        except Exception as e:
                            self.monitoring.log(f"Error executing sell order (trailing-stop, attempt {attempt + 1}/{self.retry_attempts}): {str(e)}")
                            if attempt < self.retry_attempts - 1:
                                time.sleep(self.retry_delay)
                            else:
                                self.monitoring.log("Max retries reached for trailing-stop sell order")
                                return
                else:
                    self.peak_price = max(self.peak_price, price)

            # Enter new position if signal and sufficient funds
            if (latest['signal'] == 1 and 
                self.risk_manager.initial_capital >= POSITION_SIZE_FRACTION * portfolio_value and 
                self.position == 0 and 
                self.cooldown == 0):
                trade_value = POSITION_SIZE_FRACTION * portfolio_value
                amount = (trade_value / price) * (1 - TRANSACTION_FEE_RATE)
                for attempt in range(self.retry_attempts):
                    try:
                        order = self.exchange.create_market_buy_order(SYMBOL, amount)
                        self.position = amount
                        self.active_trade = {'price': price}
                        self.peak_price = price
                        self.monitoring.log(f"Buy {amount} BTC at {price}, order: {order}")
                        break
                    except Exception as e:
                        self.monitoring.log(f"Error executing buy order (attempt {attempt + 1}/{self.retry_attempts}): {str(e)}")
                        if attempt < self.retry_attempts - 1:
                            time.sleep(self.retry_delay)
                        else:
                            self.monitoring.log("Max retries reached for buy order")
                            return

        except Exception as e:
            self.monitoring.log(f"Unexpected error in trade execution: {str(e)}")
            time.sleep(60)  # Wait before retrying

    def run(self):
        """Run live trading on a 4-hour schedule."""
        schedule.every(4).hours.do(self.trade)
        self.monitoring.log("Starting live trading loop")
        while True:
            try:
                schedule.run_pending()
                time.sleep(1)
            except Exception as e:
                self.monitoring.log(f"Error in scheduling loop: {str(e)}")
                time.sleep(60)  # Wait before retrying