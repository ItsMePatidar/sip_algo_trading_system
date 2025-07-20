"""
YFinance-Based ML System for Mutual Fund and Stock Analysis
High-quality real-time data with advanced ML predictions
"""

import yfinance as yf
import pandas as pd
import numpy as np
from flask import Flask, jsonify, request, render_template_string
from flask_cors import CORS
import logging
from datetime import datetime, timedelta
import json
from typing import Dict, List, Optional, Tuple
import warnings
import os
from dotenv import load_dotenv

load_dotenv()
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
# from sklearn.model_selection import train_test_split
import ta  # Technical Analysis Library (pip install ta)
from ta.momentum import RSIIndicator
from ta.trend import MACD

# Time Series
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

# Visualization
# import plotly.graph_objects as go
from plotly.subplots import make_subplots
# import plotly.express as px

app = Flask(__name__)
CORS(app)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Indian Market Symbols - Popular stocks and ETFs as proxies for fund analysis
INDIAN_SYMBOLS = {
    # Large Cap Stocks (Blue Chip)
    'RELIANCE.NS': 'Reliance Industries',
    'TCS.NS': 'Tata Consultancy Services',
    'HDFCBANK.NS': 'HDFC Bank',
    'INFY.NS': 'Infosys',
    'HINDUNILVR.NS': 'Hindustan Unilever',
    'ICICIBANK.NS': 'ICICI Bank',
    'KOTAKBANK.NS': 'Kotak Mahindra Bank',
    'ITC.NS': 'ITC Limited',
    'LT.NS': 'Larsen & Toubro',
    'SBIN.NS': 'State Bank of India',
    
    # Mid Cap Stocks
    'BAJFINANCE.NS': 'Bajaj Finance',
    'BAJAJFINSV.NS': 'Bajaj Finserv',
    'TECHM.NS': 'Tech Mahindra',
    'WIPRO.NS': 'Wipro',
    'HCLTECH.NS': 'HCL Technologies',
    
    # Small Cap Stocks
    'ZEEL.NS': 'Zee Entertainment',
    'IDEA.NS': 'Vodafone Idea',
    'YESBANK.NS': 'Yes Bank',
    
    # ETFs (Mutual Fund Proxies)
    'NIFTYBEES.NS': 'Nippon India ETF Nifty BeES',
    'JUNIORBEES.NS': 'Nippon India ETF Junior BeES',
    'BANKBEES.NS': 'Nippon India ETF Bank BeES',
    'LIQUIDBEES.NS': 'Nippon India ETF Liquid BeES',
    
    # Sectoral ETFs
    'ITBEES.NS': 'Nippon India ETF IT BeES',
    'PSUBNKBEES.NS': 'Nippon India ETF PSU Bank BeES',
    
    # Gold and Commodities
    'GOLDBEES.NS': 'Nippon India ETF Gold BeES',
    
    # Popular Indices
    '^NSEI': 'NIFTY 50',
    '^BSESN': 'BSE SENSEX',
    '^NSEBANK': 'NIFTY Bank',
    '^NSEIT': 'NIFTY IT',
}

# Fund categories for analysis
FUND_CATEGORIES = {
    'large_cap': ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'HINDUNILVR.NS'],
    'mid_cap': ['BAJFINANCE.NS', 'BAJAJFINSV.NS', 'TECHM.NS', 'WIPRO.NS', 'HCLTECH.NS'],
    'banking': ['HDFCBANK.NS', 'ICICIBANK.NS', 'KOTAKBANK.NS', 'SBIN.NS'],
    'technology': ['TCS.NS', 'INFY.NS', 'TECHM.NS', 'WIPRO.NS', 'HCLTECH.NS'],
    'etf': ['NIFTYBEES.NS', 'JUNIORBEES.NS', 'BANKBEES.NS', 'ITBEES.NS', 'GOLDBEES.NS'],
    'indices': ['^NSEI', '^BSESN', '^NSEBANK', '^NSEIT']
}

class YFinanceDataProcessor:
    """Advanced data processor using yfinance"""
    
    def __init__(self):
        self.cache = {}
        self.cache_duration = 300  # 5 minutes cache
    
    def get_stock_data(self, symbol: str, period: str = "2y") -> pd.DataFrame:
        """Get stock data from yfinance with caching"""
        cache_key = f"{symbol}_{period}"
        current_time = datetime.now()
        
        # Check cache
        if cache_key in self.cache:
            cached_data, cached_time = self.cache[cache_key]
            if (current_time - cached_time).seconds < self.cache_duration:
                return cached_data
        
        try:
            # Fetch data from yfinance
            # period = '5d'
            logging.info(f"Fetching data for {symbol} for period {period}")
            ticker = yf.Ticker(symbol)
            
            data = ticker.history(period=period)
            
            if data.empty:
                logging.warning(f"No data found for symbol: {symbol}")
                return pd.DataFrame()
            
            # Clean and process data
            data = data.reset_index()
            data.columns = [col.lower().replace(' ', '_') for col in data.columns]
            
            # Calculate additional metrics
            data['daily_return'] = data['close'].pct_change()
            data['volatility_20'] = data['daily_return'].rolling(20).std()
            data['sma_20'] = data['close'].rolling(20).mean()
            data['sma_50'] = data['close'].rolling(50).mean()
            data['rsi'] = RSIIndicator(close=data['close'], window=14).rsi()
            
            # Calculate MACD
            macd = MACD(close=data['close'])
            data['macd'] = macd.macd()
            data['macd_signal'] = macd.macd_signal()
            
            # Cache the data
            self.cache[cache_key] = (data, current_time)
            
            logging.info(f"Successfully fetched {len(data)} records for {symbol}")
            return data
            
        except Exception as e:
            logging.error(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_multiple_stocks(self, symbols: List[str], period: str = "1y") -> Dict[str, pd.DataFrame]:
        """Get data for multiple stocks"""
        results = {}
        for symbol in symbols:
            data = self.get_stock_data(symbol, period)
            if not data.empty:
                results[symbol] = data
        return results
    
    def calculate_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive technical indicators"""
        if data.empty:
            return data
        
        df = data.copy()
        
        try:
            # Price-based indicators
            df['sma_10'] = df['close'].rolling(10).mean()
            df['sma_50'] = df['close'].rolling(50).mean()
            df['ema_12'] = df['close'].ewm(span=12).mean()
            df['ema_26'] = df['close'].ewm(span=26).mean()
            
            # Bollinger Bands
            bb_period = 20
            df['bb_middle'] = df['close'].rolling(bb_period).mean()
            bb_std = df['close'].rolling(bb_period).std()
            df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
            df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
            
            # Momentum indicators
            df['momentum_10'] = df['close'] / df['close'].shift(10) - 1
            df['roc_10'] = ((df['close'] - df['close'].shift(10)) / df['close'].shift(10)) * 100
            
            # Volume indicators
            if 'volume' in df.columns:
                df['volume_sma'] = df['volume'].rolling(20).mean()
                df['volume_ratio'] = df['volume'] / df['volume_sma']
            
            return df
            
        except Exception as e:
            logging.error(f"Error calculating technical indicators: {e}")
            return df

class AdvancedAnalyzer:
    """Advanced financial analysis using ML"""
    
    def __init__(self):
        self.data_processor = YFinanceDataProcessor()
        self.models = {}
        
    def calculate_risk_metrics(self, data: pd.DataFrame) -> Dict:
        """Calculate comprehensive risk and performance metrics"""
        if data.empty or len(data) < 30:
            return {}
        
        try:
            returns = data['daily_return'].dropna()
            prices = data['close']
            
            # Basic performance metrics
            total_return = (prices.iloc[-1] / prices.iloc[0] - 1) * 100
            periods_per_year = 252  # Trading days
            annualized_return = ((prices.iloc[-1] / prices.iloc[0]) ** (periods_per_year / len(data)) - 1) * 100
            
            # Risk metrics
            volatility = returns.std() * np.sqrt(periods_per_year) * 100
            downside_returns = returns[returns < 0]
            downside_deviation = downside_returns.std() * np.sqrt(periods_per_year) * 100 if len(downside_returns) > 0 else 0
            
            # Sharpe and Sortino ratios (assuming 6% risk-free rate)
            risk_free_rate = 0.06
            excess_returns = returns.mean() * periods_per_year - risk_free_rate
            sharpe_ratio = excess_returns / (volatility / 100) if volatility > 0 else 0
            sortino_ratio = excess_returns / (downside_deviation / 100) if downside_deviation > 0 else 0
            
            # Maximum drawdown
            cumulative_returns = (1 + returns).cumprod()
            peak = cumulative_returns.expanding(min_periods=1).max()
            drawdown = (cumulative_returns - peak) / peak
            max_drawdown = drawdown.min() * 100
            
            # Value at Risk (95% confidence)
            var_95 = returns.quantile(0.05) * 100
            
            # Additional metrics
            positive_days = (returns > 0).sum()
            total_days = len(returns)
            win_rate = (positive_days / total_days * 100) if total_days > 0 else 0
            
            # Calmar ratio
            calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
            
            return {
                'total_return': round(total_return, 2),
                'annualized_return': round(annualized_return, 2),
                'volatility': round(volatility, 2),
                'sharpe_ratio': round(sharpe_ratio, 3),
                'sortino_ratio': round(sortino_ratio, 3),
                'max_drawdown': round(max_drawdown, 2),
                'calmar_ratio': round(calmar_ratio, 3),
                'var_95': round(var_95, 2),
                'win_rate': round(win_rate, 1),
                'downside_deviation': round(downside_deviation, 2),
                'total_trading_days': total_days
            }
            
        except Exception as e:
            logging.error(f"Error calculating risk metrics: {e}")
            return {}
    
    def ml_prediction(self, data: pd.DataFrame, forecast_days: int = 7) -> Dict:
        """Advanced ML-based price prediction"""
        if data.empty or len(data) < 60:
            return {'error': 'Insufficient data for ML prediction'}
        
        try:
            # Prepare features
            df = self.data_processor.calculate_technical_indicators(data.copy())
            
            # Feature selection
            feature_columns = [
                'sma_10', 'sma_50', 'rsi', 'macd', 'bb_width', 
                'momentum_10', 'volatility_20', 'daily_return'
            ]
            
            # Remove rows with NaN values
            df_clean = df[feature_columns + ['close']].dropna()
            
            if len(df_clean) < 30:
                return {'error': 'Insufficient clean data for prediction'}
            
            # Prepare target variable (future price)
            df_clean['target'] = df_clean['close'].shift(-forecast_days)
            df_clean = df_clean.dropna()
            
            if len(df_clean) < 20:
                return {'error': 'Insufficient data after target preparation'}
            
            # Split data
            X = df_clean[feature_columns].values
            y = df_clean['target'].values
            
            # Use recent 80% for training, 20% for testing
            split_point = int(len(X) * 0.8)
            X_train, X_test = X[:split_point], X[split_point:]
            y_train, y_test = y[:split_point], y[split_point:]
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train Random Forest model
            rf_model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            rf_model.fit(X_train_scaled, y_train)
            
            # Model performance
            y_pred = rf_model.predict(X_test_scaled)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            
            # Make future prediction
            latest_features = X[-1:].reshape(1, -1)
            latest_scaled = scaler.transform(latest_features)
            predicted_price = rf_model.predict(latest_scaled)[0]
            
            current_price = data['close'].iloc[-1]
            predicted_return = (predicted_price / current_price - 1) * 100
            
            # Confidence interval based on model error
            confidence_interval = rmse / current_price * 100
            
            return {
                'current_price': round(current_price, 2),
                'predicted_price': round(predicted_price, 2),
                'predicted_return': round(predicted_return, 2),
                'confidence_lower': round(predicted_return - confidence_interval, 2),
                'confidence_upper': round(predicted_return + confidence_interval, 2),
                'model_rmse': round(rmse, 2),
                'model_r2': round(r2, 3),
                'forecast_days': forecast_days,
                'prediction_date': (datetime.now() + timedelta(days=forecast_days)).isoformat()
            }
            
        except Exception as e:
            logging.error(f"Error in ML prediction: {e}")
            return {'error': str(e)}
    
    def time_series_forecast(self, data: pd.DataFrame, forecast_days: int = 30) -> Dict:
        """ARIMA-based time series forecasting"""
        if data.empty or len(data) < 50:
            return {'error': 'Insufficient data for time series forecasting'}
        
        try:
            # Use closing prices
            prices = data['close'].dropna()
            
            # Check stationarity
            adf_result = adfuller(prices)
            is_stationary = adf_result[1] < 0.05
            
            # Difference if not stationary
            if not is_stationary:
                prices_diff = prices.diff().dropna()
                d = 1
            else:
                prices_diff = prices
                d = 0
            
            # Fit ARIMA model with automatic order selection
            best_aic = float('inf')
            best_model = None
            best_order = None
            
            for p in range(0, 3):
                for q in range(0, 3):
                    try:
                        model = ARIMA(prices, order=(p, d, q))
                        fitted_model = model.fit()
                        if fitted_model.aic < best_aic:
                            best_aic = fitted_model.aic
                            best_model = fitted_model
                            best_order = (p, d, q)
                    except:
                        continue
            
            if best_model is None:
                return {'error': 'Could not fit ARIMA model'}
            
            # Generate forecast
            forecast_result = best_model.get_forecast(steps=forecast_days)
            forecast_values = forecast_result.predicted_mean
            confidence_intervals = forecast_result.conf_int();
            
            return {
                'forecast_values': forecast_values.tolist(),
                'confidence_lower': confidence_intervals.iloc[:, 0].tolist(),
                'confidence_upper': confidence_intervals.iloc[:, 1].tolist(),
                'model_order': best_order,
                'model_aic': round(best_aic, 2),
                'forecast_dates': [(datetime.now() + timedelta(days=i+1)).isoformat() for i in range(forecast_days)]
            }
            
        except Exception as e:
            logging.error(f"Error in time series forecasting: {e}")
            return {'error': str(e)}
    
    def calculate_score(self, risk_metrics: Dict) -> Dict:
        """Calculate investment score based on multiple factors"""
        if not risk_metrics:
            return {}
        
        try:
            # Component scores (0-100)
            return_score = min(100, max(0, risk_metrics.get('annualized_return', 0) * 3 + 50))
            risk_score = min(100, max(0, 100 - risk_metrics.get('volatility', 50)))
            sharpe_score = min(100, max(0, (risk_metrics.get('sharpe_ratio', 0) + 1) * 40))
            drawdown_score = min(100, max(0, 100 + risk_metrics.get('max_drawdown', -50) * 1.5))
            consistency_score = min(100, max(0, risk_metrics.get('win_rate', 50) * 1.5))
            
            # Weighted final score
            final_score = (
                return_score * 0.30 +
                risk_score * 0.25 +
                sharpe_score * 0.25 +
                drawdown_score * 0.10 +
                consistency_score * 0.10
            )
            
            # Investment recommendation
            if final_score >= 80:
                recommendation = "Strong Buy"
                risk_level = "Moderate"
            elif final_score >= 65:
                recommendation = "Buy"
                risk_level = "Moderate"
            elif final_score >= 50:
                recommendation = "Hold"
                risk_level = "Medium"
            elif final_score >= 35:
                recommendation = "Weak Hold"
                risk_level = "High"
            else:
                recommendation = "Avoid"
                risk_level = "Very High"
            
            return {
                'final_score': round(final_score, 1),
                'recommendation': recommendation,
                'risk_level': risk_level,
                'component_scores': {
                    'return_score': round(return_score, 1),
                    'risk_score': round(risk_score, 1),
                    'sharpe_score': round(sharpe_score, 1),
                    'drawdown_score': round(drawdown_score, 1),
                    'consistency_score': round(consistency_score, 1)
                }
            }
            
        except Exception as e:
            logging.error(f"Error calculating score: {e}")
            return {}

# Initialize analyzer
analyzer = AdvancedAnalyzer()

# Run ML pipeline at startup
from ml_model_manager import MLModelManager
DB_URL = os.getenv('DATABASE_URL')
ml_pipeline_result_cache = {}

def get_ml_pipeline_result_for_ticker(ticker):
    if ticker in ml_pipeline_result_cache:
        return ml_pipeline_result_cache[ticker]
    try:
        ml_mgr = MLModelManager(DB_URL)
        model, metrics = ml_mgr.train_model(ticker)
        result = {'ticker': ticker, **metrics}
        ml_pipeline_result_cache[ticker] = result
        return result
    except Exception as e:
        print(f"Error running ML pipeline for {ticker}: {e}")
        return None

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'data_source': 'yfinance',
        'available_symbols': len(INDIAN_SYMBOLS),
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/symbols/search', methods=['GET'])
def search_symbols():
    """Search for available symbols"""
    query = request.args.get('q', '').upper()
    limit = request.args.get('limit', 20, type=int)
    
    if not query or len(query) < 1:
        return jsonify({'error': 'Query must be at least 1 character'}), 400
    
    # Search in symbol names and descriptions
    matching_symbols = []
    for symbol, name in INDIAN_SYMBOLS.items():
        if query in symbol.upper() or query in name.upper():
            matching_symbols.append({
                'symbol': symbol,
                'name': name,
                'category': get_symbol_category(symbol)
            })
    
    return jsonify({
        'symbols': matching_symbols[:limit],
        'total_results': len(matching_symbols)
    })

@app.route('/api/symbols/categories', methods=['GET'])
def get_categories():
    """Get symbols by category"""
    return jsonify({
        'categories': {
            category: [{'symbol': symbol, 'name': INDIAN_SYMBOLS.get(symbol, symbol)} 
                      for symbol in symbols]
            for category, symbols in FUND_CATEGORIES.items()
        }
    })

@app.route('/api/analysis/<symbol>', methods=['GET'])
def analyze_symbol(symbol):
    """Comprehensive analysis for a symbol"""
    period = request.args.get('period', '2y')
    forecast_days = request.args.get('forecast_days', 7, type=int)
    
    try:
        logging.info(f"Starting analysis for symbol: {symbol}")
        
        # Get stock data
        data = analyzer.data_processor.get_stock_data(symbol, period)
        if data.empty:
            return jsonify({'error': f'No data found for symbol {symbol}'}), 404
        
        # Calculate risk metrics
        risk_metrics = analyzer.calculate_risk_metrics(data)
        
        # Calculate investment score
        investment_score = analyzer.calculate_score(risk_metrics)
        
        # ML prediction
        ml_prediction = analyzer.ml_prediction(data, forecast_days)
        
        # Time series forecast
        ts_forecast = analyzer.time_series_forecast(data, 30)
        
        # Prepare response
        result = {
            'symbol_info': {
                'symbol': symbol,
                'name': INDIAN_SYMBOLS.get(symbol, symbol),
                'category': get_symbol_category(symbol),
                'data_points': len(data),
                'date_range': {
                    'start': data['date'].min().isoformat() if 'date' in data.columns else None,
                    'end': data['date'].max().isoformat() if 'date' in data.columns else None
                },
                'latest_price': float(data['close'].iloc[-1])
            },
            'risk_metrics': risk_metrics,
            'investment_score': investment_score,
            'ml_prediction': ml_prediction,
            'time_series_forecast': ts_forecast,
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        logging.info(f"Analysis completed for symbol: {symbol}")
        return jsonify(result)
        
    except Exception as e:
        logging.error(f"Error in symbol analysis: {e}")
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

@app.route('/api/price_data/<symbol>', methods=['GET'])
def get_price_data(symbol):
    """Get price data for charting"""
    period = request.args.get('period', '1y')
    
    try:
        data = analyzer.data_processor.get_stock_data(symbol, period)
        
        if data.empty:
            return jsonify({'error': 'No data found'}), 404
        
        # Prepare data for frontend
        # logging.info(f"Preparing price data for symbol: {data['date'].dt.strftime('%Y-%m-%d').tolist() if 'date' in data.columns else []}");
        chart_data = {
            'dates': data['date'].dt.strftime('%Y-%m-%d').tolist() if 'date' in data.columns else [],
            'open': data['open'].round(2).tolist() if 'open' in data.columns else [],
            'high': data['high'].round(2).tolist() if 'high' in data.columns else [],
            'low': data['low'].round(2).tolist() if 'low' in data.columns else [],
            'close': data['close'].round(2).tolist(),
            'volume': data['volume'].tolist() if 'volume' in data.columns else [],
            'daily_returns': data['daily_return'].fillna(0).round(4).tolist(),
            'sma_20': data['sma_20'].fillna(0).round(2).tolist() if 'sma_20' in data.columns else [],
            'sma_50': data['sma_50'].fillna(0).round(2).tolist() if 'sma_50' in data.columns else []
        }
        # logging.info(f"Chart data prepared for symbol: {symbol} with {len(chart_data['dates'])} records {chart_data}");
        
        return chart_data
        
    except Exception as e:
        logging.error(f"Error getting price data: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/portfolio/compare', methods=['POST'])
def compare_portfolio():
    """Compare multiple symbols for portfolio analysis"""
    try:
        data = request.get_json()
        symbols = data.get('symbols', [])
        period = data.get('period', '1y')
        
        if not symbols or len(symbols) < 2:
            return jsonify({'error': 'At least 2 symbols required'}), 400
        
        results = {}
        for symbol in symbols:
            stock_data = analyzer.data_processor.get_stock_data(symbol, period)
            if not stock_data.empty:
                risk_metrics = analyzer.calculate_risk_metrics(stock_data)
                score = analyzer.calculate_score(risk_metrics)
                
                results[symbol] = {
                    'name': INDIAN_SYMBOLS.get(symbol, symbol),
                    'risk_metrics': risk_metrics,
                    'score': score,
                    'latest_price': float(stock_data['close'].iloc[-1])
                }
        
        # Calculate correlation matrix
        price_data = {}
        for symbol in symbols:
            stock_data = analyzer.data_processor.get_stock_data(symbol, period)
            if not stock_data.empty:
                price_data[symbol] = stock_data['daily_return'].dropna()
        
        correlation_matrix = {}
        if len(price_data) >= 2:
            df_returns = pd.DataFrame(price_data).fillna(0)
            corr_matrix = df_returns.corr()
            correlation_matrix = corr_matrix.to_dict()
        
        return jsonify({
            'comparison_results': results,
            'correlation_matrix': correlation_matrix,
            'portfolio_metrics': calculate_portfolio_metrics(results)
        })
        
    except Exception as e:
        logging.error(f"Error in portfolio comparison: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/ml_pipeline_result', methods=['GET'])
def ml_pipeline_result():
    symbol = request.args.get('symbol', None)
    if not symbol:
        return jsonify({'error': 'Symbol parameter is required'}), 400
    result = get_ml_pipeline_result_for_ticker(symbol)
    if result:
        return jsonify(result)
    else:
        return jsonify({'error': 'ML pipeline result not available'}), 404

def get_symbol_category(symbol: str) -> str:
    """Get category for a symbol"""
    for category, symbols in FUND_CATEGORIES.items():
        if symbol in symbols:
            return category
    return 'other'

def calculate_portfolio_metrics(results: Dict) -> Dict:
    """Calculate portfolio-level metrics"""
    try:
        if not results:
            return {}
        
        returns = [r['risk_metrics'].get('annualized_return', 0) for r in results.values()]
        volatilities = [r['risk_metrics'].get('volatility', 0) for r in results.values()]
        scores = [r['score'].get('final_score', 0) for r in results.values()];
        
        return {
            'avg_return': round(np.mean(returns), 2),
            'avg_volatility': round(np.mean(volatilities), 2),
            'avg_score': round(np.mean(scores), 1),
            'best_performer': max(results.keys(), key=lambda k: results[k]['score'].get('final_score', 0)),
            'lowest_risk': min(results.keys(), key=lambda k: results[k]['risk_metrics'].get('volatility', 100))
        }
    except:
        return {}

if __name__ == '__main__':
    print("🚀 Starting YFinance ML Investment System...")
    print("🔗 API Health: http://localhost:5000/api/health")
    print("📈 Sample Analysis: http://localhost:5000/api/analysis/RELIANCE.NS")
    print("🔍 Symbol Search: http://localhost:5000/api/symbols/search?q=TCS")
    
    app.run(debug=True, host='0.0.0.0', port=5000)