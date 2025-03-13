# Trading Bot Project (Version 0.6)

A Python-based trading bot for Bybit exchange using technical indicators and machine learning (XGBoost) to generate trading signals.

## Overview
This project includes scripts for fetching historical data, training an ML model, backtesting strategies, and live trading on the Bybit exchange.

## Files
- `config.py`: Configuration settings (API keys, trading parameters).
- `indicators.py`: Technical indicator calculations.
- `main.py`: Main script for fetching data, training, and backtesting.
- `ml_model.py`: Machine learning model (XGBoost) for predictions.
- `backtest.py`: Backtesting logic with HTML report generation.
- `data_handler.py`: Data fetching and loading from Bybit.
- `strategy.py`: Trading signal generation.
- `risk_management.py`: Risk management logic.
- `live_trading.py`: Live trading implementation.
- `monitoring.py`: Logging and monitoring trades.

## Setup
1. Install dependencies: `pip install ccxt pandas numpy xgboost sklearn ta schedule`
2. Create a `.env` file with your Bybit API keys (see `.env.example`).
3. Run `main.py` with appropriate arguments (e.g., `python main.py fetch`).

## Usage
- To fetch data: `python main.py fetch`
- To train the ML model: `python main.py train`
- To backtest: `python main.py backtest`

## Notes
- Ensure API keys are stored securely in a `.env` file, not in `config.py` directly.
- Backtest results are saved as HTML reports.

## Future Improvements
- Add more advanced risk management.
- Improve ML model accuracy.
- Add support for multiple exchanges.
