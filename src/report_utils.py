from src.config import INITIAL_CAPITAL

def generate_html_report(timeframe, trades, metrics):
    """Generate HTML report with trade details and summary metrics."""
    trade_rows = ""
    for trade in trades:
        trade_type_class = "buy" if trade['type'] == 'buy' else "sell"
        reason = trade.get('reason', 'signal')
        profit_loss = trade['profit_loss']
        profit_loss_display = "N/A" if trade['type'] == 'buy' else f"{profit_loss:.2f}"
        profit_loss_class = "positive" if profit_loss > 0 else "negative" if profit_loss < 0 else ""
        trade_rows += f"<tr><td>{trade['trade_number']}</td><td><span class='{trade_type_class}'>{trade['type'].upper()}</span></td>" \
                      f"<td>{trade['timestamp']}</td><td>{trade['price']:.2f}</td><td>{trade['amount']:.6f}</td>" \
                      f"<td>{trade['fee']:.2f}</td><td>{trade['portfolio_value']:.2f}</td><td>{reason}</td>" \
                      f"<td><span class='{profit_loss_class}'>{profit_loss_display}</span></td></tr>"

    html_content = f"""
    <html>
    <head>
        <title>Backtest Report - {timeframe}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f4f4f4; }}
            h1 {{ color: #333; }} h2 {{ color: #555; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 20px; background-color: #fff; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
            th, td {{ padding: 10px; border: 1px solid #ddd; text-align: left; }}
            th {{ background-color: #4CAF50; color: white; }}
            tr:nth-child(even) {{ background-color: #f2f2f2; }}
            .summary {{ margin-top: 20px; padding: 15px; background-color: #e8f5e9; border-radius: 5px; }}
            .trade-details {{ margin-top: 20px; }}
            .buy {{ color: green; font-weight: bold; }} .sell {{ color: red; font-weight: bold; }}
            .positive {{ color: green; }} .negative {{ color: red; }}
        </style>
    </head>
    <body>
        <h1>Backtest Report - {timeframe}</h1>
        <div class="summary">
            <h2>Summary</h2>
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
                <tr><td>Initial Capital</td><td>{INITIAL_CAPITAL:.2f} USDT</td></tr>
                <tr><td>Final Portfolio Value</td><td>{metrics['final_value']:.2f} USDT</td></tr>
                <tr><td>Total Return</td><td>{metrics['total_return']:.2f}%</td></tr>
                <tr><td>Sharpe Ratio (Annualized)</td><td>{metrics['sharpe_ratio']:.2f}</td></tr>
                <tr><td>Max Drawdown</td><td>{metrics['max_drawdown'] * 100:.2f}%</td></tr>
                <tr><td>Total Trades</td><td>{len(trades)} (Buys: {len(metrics['buy_trades'])}, Sells: {len(metrics['sell_trades'])})</td></tr>
                <tr><td>Win Rate</td><td>{metrics['win_rate']:.2f}%</td></tr>
                <tr><td>Total Transaction Fees</td><td>{metrics['total_fees']:.2f} USDT</td></tr>
                <tr><td>Average Holding Period (Candles)</td><td>{metrics['avg_holding_period']:.2f}</td></tr>
                <tr><td>Profit Factor</td><td>{metrics['profit_factor']:.2f}</td></tr>
                <tr><td>Max Consecutive Wins</td><td>{metrics['max_consecutive_wins']}</td></tr>
                <tr><td>Max Consecutive Losses</td><td>{metrics['max_consecutive_losses']}</td></tr>
                <tr><td>Win Rate in Trending Markets</td><td>{metrics['trending_win_rate']:.2f}% ({metrics['trend_regime_trades']['trending']} trades)</td></tr>
                <tr><td>Win Rate in Choppy Markets</td><td>{metrics['choppy_win_rate']:.2f}% ({metrics['trend_regime_trades']['choppy']} trades)</td></tr>
            </table>
        </div>
        <div class="trade-details">
            <h2>Trade Details</h2>
            <table>
                <tr><th>Trade #</th><th>Type</th><th>Timestamp</th><th>Price (USDT)</th><th>Amount (BTC)</th><th>Fee (USDT)</th><th>Portfolio Value (USDT)</th><th>Reason</th><th>Profit/Loss (USDT)</th></tr>
                {trade_rows}
            </table>
        </div>
    </body>
    </html>
    """
    return html_content