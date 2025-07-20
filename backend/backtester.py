# Backtesting logic for ML model
from ml_model_manager import MLModelManager
import numpy as np

class Backtester:
    def __init__(self, db_url=None):
        self.model_mgr = MLModelManager(db_url)

    def run_backtest(self, ticker):
        data = self.model_mgr.load_data(ticker)
        data = data.sort_values('date')
        data['prev_close'] = data['close'].shift(1)
        data = data.dropna()
        X = data[['prev_close']]
        y = data['close']
        model, mse = self.model_mgr.train_model(ticker)
        predictions = model.predict(X)
        data['predicted_close'] = predictions
        # Simple backtest: calculate returns if you bought at predicted close
        data['strategy_return'] = (data['predicted_close'].shift(-1) - data['close']) / data['close']
        data['actual_return'] = (data['close'].shift(-1) - data['close']) / data['close']
        strategy_cum_return = (1 + data['strategy_return'].fillna(0)).cumprod().iloc[-1]
        actual_cum_return = (1 + data['actual_return'].fillna(0)).cumprod().iloc[-1]
        print(f"Backtest for {ticker}:\nStrategy cumulative return: {strategy_cum_return}\nActual cumulative return: {actual_cum_return}")
        return data
