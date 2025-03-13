import pandas as pd
from config import INITIAL_CAPITAL, TRANSACTION_FEE_RATE, POSITION_SIZE_PERCENT, STOP_LOSS_PERCENT, TRAILING_STOP_PERCENT, TAKE_PROFIT_PERCENT

def backtest(df):
    """Run a backtest with signals, stop-loss, trailing stop, and take-profit."""
    cash = INITIAL_CAPITAL
    position = 0
    portfolio_values = []
    trades = []
    trade_number = 0
    active_trade = None
    peak_price = 0

    for i in range(len(df)):
        price = df['close'].iloc[i]
        timestamp = df['timestamp'].iloc[i]
        portfolio_value = cash + position * price
        portfolio_values.append(portfolio_value)

        if position > 0 and active_trade:
            if price < active_trade['price'] * (1 - STOP_LOSS_PERCENT):
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
            elif price > active_trade['price'] * (1 + TAKE_PROFIT_PERCENT):
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
            elif price < peak_price * (1 - TRAILING_STOP_PERCENT):
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
            else:
                peak_price = max(peak_price, price)

        if df['final_buy_signal'].iloc[i] and cash >= INITIAL_CAPITAL * POSITION_SIZE_PERCENT and position == 0:
            trade_number += 1
            trade_value = INITIAL_CAPITAL * POSITION_SIZE_PERCENT
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

        elif df['final_sell_signal'].iloc[i] and position > 0:
            trade_number += 1
            cash_from_sale = position * price * (1 - TRANSACTION_FEE_RATE)
            fee = position * price * TRANSACTION_FEE_RATE
            cash += cash_from_sale
            trade_info = {
                'trade_number': trade_number,
                'type': 'sell',
                'timestamp': timestamp,
                'price': price,
                'amount': position,
                'portfolio_value': portfolio_value,
                'fee': fee,
                'reason': 'signal'
            }
            trades.append(trade_info)
            position = 0
            active_trade = None
            peak_price = 0

    df['portfolio_value'] = portfolio_values
    returns = df['portfolio_value'].pct_change().dropna()
    if len(returns) > 1 and returns.std() != 0:
        sharpe_ratio = returns.mean() / returns.std() * (252 ** 0.5)
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

    trade_rows = ""
    for trade in trades:
        trade_type_class = "buy" if trade['type'] == 'buy' else "sell"
        reason = trade.get('reason', 'signal')
        trade_rows += f"<tr><td>{trade['trade_number']}</td><td><span class='{trade_type_class}'>{trade['type'].upper()}</span></td>" \
                      f"<td>{trade['timestamp']}</td><td>{trade['price']:.2f}</td><td>{trade['amount']:.6f}</td>" \
                      f"<td>{trade['fee']:.2f}</td><td>{trade['portfolio_value']:.2f}</td><td>{reason}</td></tr>"

    html_content = f"""
    <html>
    <head>
        <title>Backtest Report</title>
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
        <h1>Backtest Report</h1>
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

    with open("backtest_report.html", "w") as f:
        f.write(html_content)

    print("Backtest completed. Open 'backtest_report.html' in your browser to view the results.")

    return {
        'final_value': final_value,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'trades': trades,
        'portfolio_values': portfolio_values,
        'total_return': total_return,
        'win_rate': win_rate,
        'total_fees': total_fees
    }