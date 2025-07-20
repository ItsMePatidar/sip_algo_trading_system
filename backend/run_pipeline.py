# Sample script to run extraction, training, prediction, and backtesting
# from historical_data_manager import HistoricalDataManager
from ml_model_manager import MLModelManager
from backtester import Backtester
import os
from dotenv import load_dotenv

load_dotenv()

DB_URL = os.getenv('DATABASE_URL')
TICKER = 'RELIANCE.NS'  # Example ticker

def main():
    # Step 1: Extract and store historical data
    # print('Extracting and storing historical data...')
    # hdm = HistoricalDataManager(DB_URL)
    # hdm.fetch_and_store(TICKER)

    # Step 2: Train ML model
    print('Training ML model...')
    ml_mgr = MLModelManager(DB_URL)
    model, metrics = ml_mgr.train_model(TICKER)
    print(f"Model metrics: {metrics}")
    # Save all metrics to file for use in other modules
    with open('ml_pipeline_result.json', 'w') as f:
        import json
        json.dump({'ticker': TICKER, **metrics}, f)

    # Step 3: Backtesting
    print('Running backtest...')
    backtester = Backtester(DB_URL)
    backtester.run_backtest(TICKER)

if __name__ == '__main__':
    main()
