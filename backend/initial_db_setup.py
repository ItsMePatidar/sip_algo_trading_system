from historical_data_manager import HistoricalDataManager
from yfinance_ml_system import INDIAN_SYMBOLS
import os 
from dotenv import load_dotenv

load_dotenv()
DB_URL = os.getenv('DATABASE_URL')
if __name__ == "__main__":
    manager = HistoricalDataManager(DB_URL)
    for ticker in INDIAN_SYMBOLS.keys():
        print(f"Fetching and storing data for {ticker}")
        manager.fetch_and_store(ticker)
    print("Initial historical data setup complete.")