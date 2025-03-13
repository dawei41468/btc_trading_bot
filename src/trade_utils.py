from src.config import TRANSACTION_FEE_RATE, POSITION_SIZE_FRACTION

def execute_sell_trade(trade_number, active_trade, price, position, timestamp, portfolio_value, reason, metrics):
    """Handle selling logic for stop-loss, take-profit, or trailing stop."""
    cash_from_sale = position * price * (1 - TRANSACTION_FEE_RATE)
    fee = position * price * TRANSACTION_FEE_RATE
    profit = (price - active_trade['price']) * position - fee - active_trade['fee']
    
    if profit > 0:
        metrics['gross_profit'] += profit
        metrics['consecutive_wins'] += 1
        metrics['consecutive_losses'] = 0
        metrics['trend_regime_wins'][metrics['regime']] += 1
    else:
        metrics['gross_loss'] += abs(profit)
        metrics['consecutive_losses'] += 1
        metrics['consecutive_wins'] = 0
    metrics['max_consecutive_wins'] = max(metrics['max_consecutive_wins'], metrics['consecutive_wins'])
    metrics['max_consecutive_losses'] = max(metrics['max_consecutive_losses'], metrics['consecutive_losses'])
    metrics['trend_regime_profits'][metrics['regime']] += profit
    metrics['trend_regime_trades'][metrics['regime']] += 1
    
    holding_period = (timestamp - active_trade['timestamp']).total_seconds() / (3600 * 4)
    metrics['holding_periods'].append(holding_period)
    
    trade = {
        'trade_number': trade_number, 'type': 'sell', 'timestamp': timestamp, 'price': price,
        'amount': position, 'portfolio_value': portfolio_value, 'fee': fee, 'reason': reason,
        'profit_loss': profit
    }
    print(f"Trade {trade_number}: {reason.capitalize()} Exit at {timestamp}, Price={price:.2f}, "
          f"Profit={profit:.2f}, Portfolio Value={portfolio_value:.2f}")
    
    return trade, cash_from_sale, 4 if profit < 0 else 2

def execute_buy_trade(trade_number, signal, price, portfolio_value, cash, timestamp, regime):
    """Handle buying logic with volatility-adjusted sizing."""
    if signal != 1 or cash < POSITION_SIZE_FRACTION * portfolio_value * 0.50:
        return None, 0, cash
    
    trade_value = POSITION_SIZE_FRACTION * portfolio_value if regime == 'trending' else 0.50 * portfolio_value
    if cash < trade_value:
        return None, 0, cash
    
    amount_to_buy = (trade_value / price) * (1 - TRANSACTION_FEE_RATE)
    trade_info = {
        'trade_number': trade_number, 'type': 'buy', 'timestamp': timestamp, 'price': price,
        'amount': amount_to_buy, 'portfolio_value': portfolio_value,
        'fee': amount_to_buy * price * TRANSACTION_FEE_RATE, 'profit_loss': 0
    }
    print(f"Trade {trade_number}: Buy at {timestamp}, Price={price:.2f}, "
          f"Amount={amount_to_buy:.6f}, Portfolio Value={portfolio_value:.2f}")
    return trade_info, amount_to_buy, cash - trade_value