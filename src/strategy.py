import pandas as pd
from config import ML_FEATURES

class TradingStrategy:
    def __init__(self, ml_model):
        self.ml_model = ml_model

    def generate_signals(self, df):
        print("Generating Signals...")
        df['buy_signal'] = (df['RSI'] < 40) & (df['MACD'] > df['MACD_signal'])
        df['sell_signal'] = (df['RSI'] > 60) & (df['MACD'] < df['MACD_signal'])
        print(f"Buy Signal Count: {df['buy_signal'].sum()}, Sell Signal Count: {df['sell_signal'].sum()}")

        if self.ml_model:
            X = df[ML_FEATURES]
            df['ml_pred'] = self.ml_model.predict(X)
            df['final_buy_signal'] = df['buy_signal'] & (df['ml_pred'] == 1)
            df['final_sell_signal'] = df['sell_signal'] & (df['ml_pred'] == 0)
            print(f"Final Buy Signal Count: {df['final_buy_signal'].sum()}, Final Sell Signal Count: {df['final_sell_signal'].sum()}")
        else:
            df['final_buy_signal'] = df['buy_signal']
            df['final_sell_signal'] = df['sell_signal']
            print("No ML model provided. Using base signals only.")

        return df