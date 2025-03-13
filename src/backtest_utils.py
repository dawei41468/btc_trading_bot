import pandas as pd
from src.data_handler import DataHandler
from src.indicators import calculate_indicators
from src.ml_model import MLModel
from src.trade_utils import execute_sell_trade, execute_buy_trade
from src.report_utils import generate_html_report
from src.config import INITIAL_CAPITAL, ML_FEATURES, TRANSACTION_FEE_RATE, POSITION_SIZE_FRACTION, TRAILING_STOP_PERCENT
import xgboost as xgb

def calculate_metrics(df, trades, trend_metrics):
    """Calculate performance metrics from portfolio values and trades."""
    returns = df['portfolio_value'].pct_change().dropna()
    sharpe_ratio = returns.mean() / returns.std() * (252 ** 0.5) if len(returns) > 1 and returns.std() != 0 else 0.0
    if sharpe_ratio == 0:
        print("Warning: Sharpe Ratio set to 0 due to insufficient returns data or zero standard deviation.")
    
    cumulative_max = df['portfolio_value'].cummax()
    drawdowns = (cumulative_max - df['portfolio_value']) / cumulative_max
    max_drawdown = drawdowns.max() if not drawdowns.empty else 0.0
    final_value = df['portfolio_value'].iloc[-1]
    total_return = (final_value - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    total_fees = sum(trade['fee'] for trade in trades)
    
    buy_trades = [t for t in trades if t['type'] == 'buy']
    sell_trades = [t for t in trades if t['type'] == 'sell']
    profitable_trades = sum(1 for i, sell in enumerate(sell_trades) if i < len(buy_trades) and sell['price'] > buy_trades[i]['price'])
    win_rate = (profitable_trades / len(sell_trades) * 100) if sell_trades else 0
    
    avg_holding_period = sum(trend_metrics['holding_periods']) / len(trend_metrics['holding_periods']) if trend_metrics['holding_periods'] else 0
    profit_factor = trend_metrics['gross_profit'] / trend_metrics['gross_loss'] if trend_metrics['gross_loss'] != 0 else float('inf')
    trending_win_rate = (trend_metrics['trend_regime_wins']['trending'] / trend_metrics['trend_regime_trades']['trending'] * 100) if trend_metrics['trend_regime_trades']['trending'] > 0 else 0
    choppy_win_rate = (trend_metrics['trend_regime_wins']['choppy'] / trend_metrics['trend_regime_trades']['choppy'] * 100) if trend_metrics['trend_regime_trades']['choppy'] > 0 else 0

    return {
        'sharpe_ratio': sharpe_ratio, 'max_drawdown': max_drawdown, 'total_return': total_return,
        'total_fees': total_fees, 'buy_trades': buy_trades, 'sell_trades': sell_trades, 'win_rate': win_rate,
        'avg_holding_period': avg_holding_period, 'profit_factor': profit_factor,
        'max_consecutive_wins': trend_metrics['max_consecutive_wins'], 'max_consecutive_losses': trend_metrics['max_consecutive_losses'],
        'trending_win_rate': trending_win_rate, 'choppy_win_rate': choppy_win_rate,
        'final_value': final_value,
        'trend_regime_trades': trend_metrics['trend_regime_trades']
    }

def backtest(timeframe='4h'):
    """Run a backtest with ML signals and ATR-based exits."""
    handler = DataHandler()
    df = handler.load_historical_data(timeframe)
    df = calculate_indicators(df)
    model = MLModel()
    X = df[ML_FEATURES]
    df['pred_prob'] = model.predict(xgb.DMatrix(X))
    df['signal'] = (df['pred_prob'] > 0.65).astype(int)
    df['trend_regime'] = (df['SMA50'] > df['SMA200']).astype(int)

    cash = INITIAL_CAPITAL
    position = 0
    portfolio_values = []
    trades = []
    trade_number = 0
    active_trade = None
    peak_price = 0
    cooldown = 0

    trend_metrics = {
        'gross_profit': 0, 'gross_loss': 0, 'consecutive_wins': 0, 'consecutive_losses': 0,
        'max_consecutive_wins': 0, 'max_consecutive_losses': 0, 'holding_periods': [],
        'trend_regime_profits': {'trending': 0, 'choppy': 0}, 'trend_regime_trades': {'trending': 0, 'choppy': 0},
        'trend_regime_wins': {'trending': 0, 'choppy': 0}, 'regime': None
    }

    for i in range(len(df)):
        price = df['close'].iloc[i]
        timestamp = df['timestamp'].iloc[i]
        signal = df['signal'].iloc[i]
        atr = df['ATR'].iloc[i]
        portfolio_value = cash + position * price
        regime = 'trending' if df['trend_regime'].iloc[i] == 1 else 'choppy'
        trend_metrics['regime'] = regime
        portfolio_values.append(portfolio_value)

        if i % 100 == 0:
            print(f"Candle {i}: Timestamp={timestamp}, Price={price:.2f}, Portfolio Value={portfolio_value:.2f}, "
                  f"Position={position:.6f}, Signal={signal}, ATR={atr:.2f}, Regime={regime}")

        if cooldown > 0:
            cooldown -= 1
            continue

        if position > 0 and active_trade:
            stop_loss_multiplier = 0.75 if regime == 'choppy' else 1.0
            if price < active_trade['price'] - atr * stop_loss_multiplier:
                trade, cash_gain, cooldown = execute_sell_trade(trade_number + 1, active_trade, price, position, timestamp, portfolio_value, 'stop-loss', trend_metrics)
                trades.append(trade)
                cash += cash_gain
                trade_number += 1
                position = 0
                active_trade = None
                peak_price = 0
            elif price > active_trade['price'] + 5 * atr:
                trade, cash_gain, cooldown = execute_sell_trade(trade_number + 1, active_trade, price, position, timestamp, portfolio_value, 'take-profit', trend_metrics)
                trades.append(trade)
                cash += cash_gain
                trade_number += 1
                position = 0
                active_trade = None
                peak_price = 0
            elif price < peak_price * (1 - TRAILING_STOP_PERCENT):
                trade, cash_gain, cooldown = execute_sell_trade(trade_number + 1, active_trade, price, position, timestamp, portfolio_value, 'trailing-stop', trend_metrics)
                trades.append(trade)
                cash += cash_gain
                trade_number += 1
                position = 0
                active_trade = None
                peak_price = 0
            else:
                peak_price = max(peak_price, price)
        else:
            trade_info, amount, new_cash = execute_buy_trade(trade_number + 1, signal, price, portfolio_value, cash, timestamp, regime)
            if trade_info:
                trades.append(trade_info)
                position = amount
                cash = new_cash
                active_trade = trade_info
                peak_price = price
                trade_number += 1

    df['portfolio_value'] = portfolio_values
    metrics = calculate_metrics(df, trades, trend_metrics)
    html_content = generate_html_report(timeframe, trades, metrics)
    
    with open(f"backtest_report_{timeframe}.html", "w") as f:
        f.write(html_content)
    print(f"Backtest completed on {timeframe}. Open 'backtest_report_{timeframe}.html' in your browser to view the results.")