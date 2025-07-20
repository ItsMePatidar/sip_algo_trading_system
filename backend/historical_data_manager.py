# Extracts historical data using yfinance and stores in Postgres
import yfinance as yf
import pandas as pd
from db_manager import DatabaseManager, HistoricalPrice
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
class HistoricalDataManager:
    def __init__(self, db_url=None):
        print("Initializing HistoricalDataManager with url:", DATABASE_URL)
        self.db = DatabaseManager(DATABASE_URL)

    def fetch_and_store(self, ticker):
        end = datetime.today()
        start = end - timedelta(days=200*1)
        df = yf.download(ticker, start=start, end=end)
        df.reset_index(inplace=True)
        # Prepare DataFrame for DB
        df['ticker'] = ticker
        df['date'] = df['Date'].apply(lambda x: x.strftime('%Y-%m-%d') if hasattr(x, 'strftime') else str(x))
        df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.date
        df = df[['ticker', 'date', 'Open', 'High', 'Low', 'Close', 'Volume']]
        df.columns = ['ticker', 'date', 'open', 'high', 'low', 'close', 'volume']
        print(f"Storing data for {ticker} from {start} to {end}")
        df.to_sql('historical_prices', self.db.engine, if_exists='append', index=False)
        
