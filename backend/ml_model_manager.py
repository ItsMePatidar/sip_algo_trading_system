# ML model building, training, and deployment using historical data
import pandas as pd
from db_manager import DatabaseManager, HistoricalPrice
from sqlalchemy.orm import sessionmaker
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

class MLModelManager:
    def __init__(self, db_url=None):
        self.db = DatabaseManager(db_url)

    def load_data(self, ticker):
        session = self.db.get_session()
        query = session.query(HistoricalPrice).filter_by(ticker=ticker)
        data = pd.read_sql(query.statement, session.bind)
        session.close()
        return data

    def train_model(self, ticker):
        data = self.load_data(ticker)
        data = data.sort_values('date')
        # Simple feature: previous close
        data['prev_close'] = data['close'].shift(1)
        data = data.dropna()
        X = data[['prev_close']]
        y = data['close']
        model = LinearRegression()
        model.fit(X, y)
        predictions = model.predict(X)
        mse = mean_squared_error(y, predictions)
        rmse = np.sqrt(mse)
        r2 = model.score(X, y)
        mae = np.mean(np.abs(y - predictions))
        mape = np.mean(np.abs((y - predictions) / y)) * 100
        print(f"Model trained for {ticker}. MSE: {mse}, RMSE: {rmse}, R2: {r2}, MAE: {mae}, MAPE: {mape}")
        return model, {
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'mae': mae,
            'mape': mape
        }

    def predict(self, model, last_close):
        return model.predict(np.array([[last_close]]))[0]
