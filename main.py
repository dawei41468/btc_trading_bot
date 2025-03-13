import sys
import pandas as pd
from data_handler import DataHandler
from indicators import calculate_indicators
from ml_model import MLModel
from config import (
    TIMEFRAME, INITIAL_CAPITAL, ML_FEATURES, TRANSACTION_FEE_RATE,
    POSITION_SIZE_FRACTION, TRAILING_STOP_PERCENT,
    ATR_TAKE_PROFIT_UPTREND, ATR_TAKE_PROFIT_NON_TREND, ATR_TAKE_PROFIT_INTERMEDIATE
)
import xgboost as xgb

def fetch_and_save_all_timeframes():
    """Fetch and save historical data for multiple timeframes."""
    handler = DataHandler()
    timeframes = ['4h', '1h', '15m']
    for tf in timeframes:
        handler.fetch_historical_data(tf)

def train(timeframe=TIMEFRAME):
    """Train the ML model on historical data."""
    handler = DataHandler()
    df = handler.load_historical_data(timeframe)
    if df.empty:
        print(f"No {timeframe} data available for training.")
        return
    df = calculate_indicators(df)
    model = MLModel()
    accuracy = model.train(df)
    print(f"ML Model Trained on {timeframe} data. Accuracy: {accuracy:.2f}")

def backtest(timeframe=TIMEFRAME, period_name=None, stress_type=None):
    """Run a backtest with ML signals and hybrid ATR-based exits."""
    # Load and prepare data
    handler = DataHandler()
    if stress_type:
        df = handler.load_stress_data(timeframe, stress_type)
    else:
        df = handler.load_historical_data(timeframe, period_name=period_name)
    if df.empty:
        print(f"No {timeframe} data available for period {period_name} or stress type {stress_type}.")
        return
    df = calculate_indicators(df)
    model = MLModel()
    X = df[ML_FEATURES]
    df['pred_prob'] = model.predict(xgb.DMatrix(X))
    df['signal'] = (df['pred_prob'] > 0.65).astype(int)

    # Initialize backtest variables
    cash = INITIAL_CAPITAL
    position = 0
    portfolio_values = []
    trades = []
    trade_number = 0
    active_trade = None
    peak_price = 0
    cooldown = 0

    # Iterate through each candle
    for i in range(len(df)):
        price = df['close'].iloc[i]
        timestamp = df['timestamp'].iloc[i]
        signal = df['signal'].iloc[i]
        atr = df['ATR'].iloc[i]
        ema20 = df['EMA20'].iloc[i]
        ema50 = df['EMA50'].iloc[i]
        ema200 = df['EMA200'].iloc[i]
        portfolio_value = cash + position * price
        portfolio_values.append(portfolio_value)

        # Hybrid trend detection for ATR take-profit multiplier
        if ema50 > ema200:  # Long-term uptrend
            atr_take_profit_multiplier = ATR_TAKE_PROFIT_UPTREND
        elif ema20 <= ema50:  # Short-term bear/sideways
            atr_take_profit_multiplier = ATR_TAKE_PROFIT_NON_TREND
        else:  # Short-term uptrend within long-term non-uptrend
            atr_take_profit_multiplier = ATR_TAKE_PROFIT_INTERMEDIATE

        if cooldown > 0:
            cooldown -= 1

        if position > 0 and active_trade:
            if price < active_trade['price'] - atr:  # 1x ATR stop-loss
                trade_number += 1
                cash_from_sale = position * price * (1 - TRANSACTION_FEE_RATE)
                fee = position * price * TRANSACTION_FEE_RATE
                cash += cash_from_sale
                trades.append({
                    'trade_number': trade_number,
                    'type': 'sell',
                    'timestamp': timestamp,
                    'price': price,
                    'amount': position,
                    'portfolio_value': portfolio_value,
                    'fee': fee,
                    'reason': 'stop-loss'
                })
                position = 0
                active_trade = None
                peak_price = 0
                cooldown = 2
            elif price > active_trade['price'] + atr_take_profit_multiplier * atr:  # Hybrid ATR take-profit
                trade_number += 1
                cash_from_sale = position * price * (1 - TRANSACTION_FEE_RATE)
                fee = position * price * TRANSACTION_FEE_RATE
                cash += cash_from_sale
                trades.append({
                    'trade_number': trade_number,
                    'type': 'sell',
                    'timestamp': timestamp,
                    'price': price,
                    'amount': position,
                    'portfolio_value': portfolio_value,
                    'fee': fee,
                    'reason': 'take-profit'
                })
                position = 0
                active_trade = None
                peak_price = 0
                cooldown = 2
            elif price < peak_price * (1 - TRAILING_STOP_PERCENT):  # 3% trailing stop
                trade_number += 1
                cash_from_sale = position * price * (1 - TRANSACTION_FEE_RATE)
                fee = position * price * TRANSACTION_FEE_RATE
                cash += cash_from_sale
                trades.append({
                    'trade_number': trade_number,
                    'type': 'sell',
                    'timestamp': timestamp,
                    'price': price,
                    'amount': position,
                    'portfolio_value': portfolio_value,
                    'fee': fee,
                    'reason': 'trailing-stop'
                })
                position = 0
                active_trade = None
                peak_price = 0
                cooldown = 2
            else:
                peak_price = max(peak_price, price)

        if signal == 1 and cash >= POSITION_SIZE_FRACTION * portfolio_value and position == 0 and cooldown == 0:
            trade_number += 1
            trade_value = POSITION_SIZE_FRACTION * portfolio_value
            amount_to_buy = (trade_value / price) * (1 - TRANSACTION_FEE_RATE)
            position = amount_to_buy
            cash -= trade_value
            trade_info = {
                'trade_number': trade_number,
                'type': 'buy',
                'timestamp': timestamp,
                'price': price,
                'amount': amount_to_buy,
                'portfolio_value': portfolio_value,
                'fee': amount_to_buy * price * TRANSACTION_FEE_RATE
            }
            trades.append(trade_info)
            active_trade = trade_info
            peak_price = price

    # Calculate performance metrics
    df['portfolio_value'] = portfolio_values
    returns = df['portfolio_value'].pct_change().dropna()
    if len(returns) > 1 and returns.std() != 0:
        sharpe_ratio = returns.mean() / returns.std() * (252 ** 0.5)  # Annualized Sharpe (252 4h periods/year)
    else:
        sharpe_ratio = 0.0
        print("Warning: Sharpe Ratio set to 0 due to insufficient returns data or zero standard deviation.")
    cumulative_max = df['portfolio_value'].cummax()
    drawdowns = (cumulative_max - df['portfolio_value']) / cumulative_max
    max_drawdown = drawdowns.max() if not drawdowns.empty else 0.0
    final_value = df['portfolio_value'].iloc[-1]
    total_return = (final_value - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    total_fees = sum(trade['fee'] for trade in trades)
    buy_trades = [trade for trade in trades if trade['type'] == 'buy']
    sell_trades = [trade for trade in trades if trade['type'] == 'sell']
    profitable_trades = 0
    for i, sell_trade in enumerate(sell_trades):
        if i < len(buy_trades):
            buy_price = buy_trades[i]['price']
            sell_price = sell_trade['price']
            if sell_price > buy_price:
                profitable_trades += 1
    win_rate = (profitable_trades / len(sell_trades) * 100) if sell_trades else 0

    # Generate HTML trade rows
    trade_rows = ""
    for trade in trades:
        trade_type_class = "buy" if trade['type'] == 'buy' else "sell"
        reason = trade.get('reason', 'signal')
        trade_rows += f"<tr><td>{trade['trade_number']}</td><td><span class='{trade_type_class}'>{trade['type'].upper()}</span></td>" \
                      f"<td>{trade['timestamp']}</td><td>{trade['price']:.2f}</td><td>{trade['amount']:.6f}</td>" \
                      f"<td>{trade['fee']:.2f}</td><td>{trade['portfolio_value']:.2f}</td><td>{reason}</td></tr>"

    # Generate HTML report
    html_content = f"""
    <html>
    <head>
        <title>Backtest Report - {timeframe}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f4f4f4; }}
            h1 {{ color: #333; }}
            h2 {{ color: #555; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 20px; background-color: #fff; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
            th, td {{ padding: 10px; border: 1px solid #ddd; text-align: left; }}
            th {{ background-color: #4CAF50; color: white; }}
            tr:nth-child(even) {{ background-color: #f2f2f2; }}
            .summary {{ margin-top: 20px; padding: 15px; background-color: #e8f5e9; border-radius: 5px; }}
            .trade-details {{ margin-top: 20px; }}
            .buy {{ color: green; font-weight: bold; }}
            .sell {{ color: red; font-weight: bold; }}
        </style>
    </head>
    <body>
        <h1>Backtest Report - {timeframe}</h1>
        <div class="summary">
            <h2>Summary</h2>
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
                <tr><td>Initial Capital</td><td>{INITIAL_CAPITAL:.2f} USDT</td></tr>
                <tr><td>Final Portfolio Value</td><td>{final_value:.2f} USDT</td></tr>
                <tr><td>Total Return</td><td>{total_return:.2f}%</td></tr>
                <tr><td>Sharpe Ratio (Annualized)</td><td>{sharpe_ratio:.2f}</td></tr>
                <tr><td>Max Drawdown</td><td>{max_drawdown * 100:.2f}%</td></tr>
                <tr><td>Total Trades</td><td>{len(trades)} (Buys: {len(buy_trades)}, Sells: {len(sell_trades)})</td></tr>
                <tr><td>Win Rate</td><td>{win_rate:.2f}%</td></tr>
                <tr><td>Total Transaction Fees</td><td>{total_fees:.2f} USDT</td></tr>
            </table>
        </div>
        <div class="trade-details">
            <h2>Trade Details</h2>
            <table>
                <tr><th>Trade #</th><th>Type</th><th>Timestamp</th><th>Price (USDT)</th><th>Amount (BTC)</th><th>Fee (USDT)</th><th>Portfolio Value (USDT)</th><th>Reason</th></tr>
                {trade_rows}
            </table>
        </div>
    </body>
    </html>
    """

    report_name = f"backtest_report_{timeframe}_{stress_type or period_name or 'full'}"
    with open(f"{report_name}.html", "w") as f:
        f.write(html_content)
    print(f"Backtest completed on {timeframe} for {stress_type or period_name or 'full dataset'}. Open '{report_name}.html' to view results.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py [fetch|train|backtest] [optional: timeframe] [optional: period_name] [optional: stress_type]")
        sys.exit(1)
    command = sys.argv[1].lower()
    if command == "fetch":
        fetch_and_save_all_timeframes()
    elif command == "train":
        timeframe = sys.argv[2] if len(sys.argv) > 2 else TIMEFRAME
        train(timeframe)
    elif command == "backtest":
        timeframe = sys.argv[2] if len(sys.argv) > 2 else TIMEFRAME
        period_name = sys.argv[3] if len(sys.argv) > 3 else None
        stress_type = sys.argv[4] if len(sys.argv) > 4 else None
        backtest(timeframe, period_name, stress_type)
    else:
        print(f"Unknown command: {command}. Use 'fetch', 'train', or 'backtest'.")