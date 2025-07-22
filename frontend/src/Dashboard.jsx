import React, { useState, useEffect } from 'react';
import Plot from 'react-plotly.js';
import { FaSearch, FaChartLine, FaRobot, FaBalanceScale, FaSyncAlt } from 'react-icons/fa';

const categoryList = [
  { key: 'large_cap', label: 'Large Cap' },
  { key: 'banking', label: 'Banking' },
  { key: 'technology', label: 'Technology' },
  { key: 'etf', label: 'ETFs' },
  { key: 'indices', label: 'Indices' },
];

const timeWindows = [
  { label: '1 Month', value: '1mo' },
  { label: '3 Months', value: '3mo' },
  { label: '6 Months', value: '6mo' },
  { label: '1 Year', value: '1y' },
  { label: '2 Years', value: '2y' },
];

function Dashboard() {
  const [searchQuery, setSearchQuery] = useState('');
  const [symbols, setSymbols] = useState([]);
  const [category, setCategory] = useState('large_cap');
  const [selectedSymbols, setSelectedSymbols] = useState([]);
  const [currentSymbol, setCurrentSymbol] = useState(null);
  const [analysis, setAnalysis] = useState(null);
  const [priceData, setPriceData] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [comparisonResults, setComparisonResults] = useState(null);
  const [portfolioMetrics, setPortfolioMetrics] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [selectedTimeWindow, setSelectedTimeWindow] = useState('1y');
  const [mlPipelineResult, setMlPipelineResult] = useState(null);

  useEffect(() => {
    fetchCategory(category);
  }, [category]);

  const fetchCategory = async (cat) => {
    setLoading(true);
    try {
      const res = await fetch('https://sip-algo-trading-system.onrender.com/api/symbols/categories');
      const data = await res.json();
      setSymbols(data.categories[cat] || []);
      setLoading(false);
    } catch (e) {
      setError('Failed to load categories');
      setLoading(false);
    }
  };

  const searchSymbols = async (query) => {
    setLoading(true);
    try {
      const res = await fetch(`https://sip-algo-trading-system.onrender.com/api/symbols/search?q=${query}&limit=10`);
      const data = await res.json();
      setSymbols(data.symbols || []);
      setLoading(false);
    } catch (e) {
      setError('Search failed');
      setLoading(false);
    }
  };

  const fetchPriceData = async (symbol, period = selectedTimeWindow) => {
    try {
      const priceRes = await fetch(`https://sip-algo-trading-system.onrender.com/api/price_data/${symbol}?period=${period}`);
      const priceData = await priceRes.json();
      setPriceData(priceData);
    } catch (e) {
      setPriceData(null);
    }
  };

  const selectSymbol = async (symbol, name) => {
    setCurrentSymbol(symbol);
    setAnalysis(null);
    setPriceData(null);
    setPrediction(null);
    setError(null);
    setLoading(true);
    // Toggle selection for portfolio
    setSelectedSymbols((prev) =>
      prev.includes(symbol) ? prev.filter((s) => s !== symbol) : [...prev, symbol]
    );
    try {
      // Fetch ML pipeline result only when a symbol is selected
      const mlRes = await fetch(`https://sip-algo-trading-system.onrender.com/api/ml_pipeline_result?symbol=${symbol}`);
      const mlData = await mlRes.json();
      setMlPipelineResult(mlData);
      const analysisRes = await fetch(`https://sip-algo-trading-system.onrender.com/api/analysis/${symbol}?forecast_days=7`);
      const analysisData = await analysisRes.json();
      if (analysisRes.ok) {
        setAnalysis(analysisData);
        setPrediction({
          ml: analysisData.ml_prediction,
          ts: analysisData.time_series_forecast,
        });
        await fetchPriceData(symbol, selectedTimeWindow);
      } else {
        setError(analysisData.error);
      }
    } catch (e) {
      setError('Analysis failed');
    }
    setLoading(false);
  };

  const compareSelected = async () => {
    if (selectedSymbols.length < 2) {
      setError('Please select at least 2 symbols for comparison');
      return;
    }
    setLoading(true);
    try {
      const res = await fetch('https://sip-algo-trading-system.onrender.com/api/portfolio/compare', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ symbols: selectedSymbols, period: '1y' }),
      });
      const data = await res.json();
      setComparisonResults(data.comparison_results);
      setPortfolioMetrics(data.portfolio_metrics);
      setLoading(false);
    } catch (e) {
      setError('Comparison failed');
      setLoading(false);
    }
  };

  useEffect(() => {
    if (currentSymbol) {
      fetchPriceData(currentSymbol, selectedTimeWindow);
    }
    // eslint-disable-next-line
  }, [selectedTimeWindow]);

  return (
    <div className="algo-dashboard" style={{ background: '#181c24', minHeight: '100vh', color: '#e1e5e9', fontFamily: 'Inter, Arial, sans-serif' }}>
      <div className="header" style={{ padding: '32px 0 16px 0', textAlign: 'center', background: '#23293a', boxShadow: '0 2px 8px #0002' }}>
        <h1 style={{ fontSize: 36, fontWeight: 700, color: '#4caf50', marginBottom: 8 }}><FaChartLine style={{ marginRight: 10 }} /> Algo Trading Dashboard</h1>
        <p style={{ fontSize: 18, color: '#b0b8c1' }}>Real-time analysis & ML predictions powered by YFinance</p>
      </div>

      {/* Symbol Search & Analysis - full width on top */}
      <div className="card" style={{ background: '#23293a', borderRadius: 16, boxShadow: '0 2px 12px #0003', padding: 24, margin: '32px 5vw 0 5vw' }}>
        <h3 style={{ color: '#4caf50', marginBottom: 16 }}><FaSearch style={{ marginRight: 8 }} /> Symbol Search & Analysis</h3>
        <div className="search-section" style={{ display: 'flex', gap: 12, marginBottom: 16 }}>
          <input
            type="text"
            className="search-box"
            style={{ flex: 1, padding: '10px 16px', borderRadius: 8, border: 'none', background: '#181c24', color: '#e1e5e9', fontSize: 16 }}
            placeholder="Search stocks/ETFs (e.g., RELIANCE.NS, TCS.NS, NIFTY)"
            value={searchQuery}
            onChange={(e) => {
              setSearchQuery(e.target.value);
              if (e.target.value.length >= 1) searchSymbols(e.target.value);
              else fetchCategory(category);
            }}
          />
          <button onClick={() => searchSymbols(searchQuery)} style={{ background: '#4caf50', color: '#fff', border: 'none', borderRadius: 8, padding: '0 16px', fontSize: 18 }}><FaSyncAlt /></button>
        </div>
        <div className="category-tabs" style={{ display: 'flex', gap: 8, marginBottom: 16 }}>
          {categoryList.map((cat) => (
            <button
              key={cat.key}
              className={`tab${category === cat.key ? ' active' : ''}`}
              style={{
                background: category === cat.key ? '#4caf50' : '#181c24',
                color: category === cat.key ? '#fff' : '#b0b8c1',
                border: 'none',
                borderRadius: 8,
                padding: '8px 18px',
                fontWeight: 600,
                fontSize: 15,
                cursor: 'pointer',
                boxShadow: category === cat.key ? '0 2px 8px #4caf5040' : 'none',
              }}
              onClick={() => {
                setCategory(cat.key);
                setSearchQuery('');
              }}
            >
              {cat.label}
            </button>
          ))}
        </div>
        <div className="symbol-grid" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(120px, 1fr))', gap: 12 }}>
          {symbols.map((symbol) => (
            <div
              key={symbol.symbol}
              className={`symbol-item${selectedSymbols.includes(symbol.symbol) ? ' selected' : ''}`}
              style={{
                background: selectedSymbols.includes(symbol.symbol) ? '#4caf50' : '#181c24',
                color: selectedSymbols.includes(symbol.symbol) ? '#fff' : '#e1e5e9',
                borderRadius: 8,
                padding: '12px 8px',
                cursor: 'pointer',
                boxShadow: selectedSymbols.includes(symbol.symbol) ? '0 2px 8px #4caf5040' : 'none',
                border: '1px solid #23293a',
                textAlign: 'center',
              }}
              onClick={() => selectSymbol(symbol.symbol, symbol.name)}
            >
              <strong>{symbol.symbol}</strong><br />
              <small style={{ color: '#b0b8c1' }}>{symbol.name}</small>
            </div>
          ))}
        </div>
      </div>

      {/* AI Predictions & ML Pipeline side by side */}
      <div className="main-grid" style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 32, padding: '32px 5vw' }}>
        <div className="card" style={{ background: '#23293a', borderRadius: 16, boxShadow: '0 2px 12px #0003', padding: 24 }}>
          <h3 style={{ color: '#4caf50', marginBottom: 16 }}><FaRobot style={{ marginRight: 8 }} /> AI Predictions & Forecasts</h3>
          <div id="predictionSection">
            {prediction && (
              <div>
                <h4 style={{ color: '#fff', marginBottom: 8 }}>🤖 AI Predictions</h4>
                {!prediction.ml?.error && (
                  <div className="prediction-box" style={{ background: '#181c24', borderRadius: 8, padding: 16, marginBottom: 16 }}>
                    <h5 style={{ color: '#4caf50', marginBottom: 8 }}>7-Day ML Prediction</h5>
                    <p><strong>Current Price:</strong> ₹{prediction.ml.current_price}</p>
                    <p><strong>Predicted Price:</strong> ₹{prediction.ml.predicted_price}</p>
                    <p><strong>Expected Return:</strong> <span style={{ color: prediction.ml.predicted_return >= 0 ? '#4caf50' : '#f44336', fontWeight: 'bold' }}>{prediction.ml.predicted_return}%</span></p>
                    <p><strong>Confidence Range:</strong> {prediction.ml.confidence_lower}% to {prediction.ml.confidence_upper}%</p>
                    <p><strong>Model Accuracy (R²):</strong> {prediction.ml.model_r2}</p>
                  </div>
                )}
                {!prediction.ts?.error && (
                  <div className="prediction-box" style={{ background: '#181c24', borderRadius: 8, padding: 16 }}>
                    <h5 style={{ color: '#4caf50', marginBottom: 8 }}>30-Day ARIMA Forecast</h5>
                    <p><strong>Model:</strong> ARIMA{JSON.stringify(prediction.ts.model_order)}</p>
                    <p><strong>AIC Score:</strong> {prediction.ts.model_aic}</p>
                    <p><strong>Forecast Range:</strong> ₹{Math.min(...prediction.ts.forecast_values).toFixed(2)} - ₹{Math.max(...prediction.ts.forecast_values).toFixed(2)}</p>
                  </div>
                )}
              </div>
            )}
            {!prediction && <p style={{ color: '#666', textAlign: 'center', padding: 40 }}>Select an asset to view AI predictions</p>}
          </div>
        </div>
        <div className="card" style={{ background: '#23293a', borderRadius: 16, boxShadow: '0 2px 12px #0003', padding: 24 }}>
          <h3 style={{ color: '#4caf50', marginBottom: 16 }}><FaRobot style={{ marginRight: 8 }} /> ML Pipeline Model Performance</h3>
          {mlPipelineResult && !mlPipelineResult.error ? (
            <div>
              <p><strong>Ticker:</strong> {mlPipelineResult.ticker}</p>
              <p><strong>MSE:</strong> {mlPipelineResult.mse}</p>
              <p><strong>RMSE:</strong> {mlPipelineResult.rmse}</p>
              <p><strong>R²:</strong> {mlPipelineResult.r2}</p>
              <p><strong>MAE:</strong> {mlPipelineResult.mae}</p>
              <p><strong>MAPE:</strong> {mlPipelineResult.mape}%</p>
            </div>
          ) : (
            <p style={{ color: '#f44336' }}>ML pipeline result not available.</p>
          )}
        </div>
      </div>

      {/* Price Chart & Technical Analysis - full width */}
      <div className="card" style={{ background: '#23293a', borderRadius: 16, boxShadow: '0 2px 12px #0003', padding: 24, margin: '32px 5vw' }}>
        <h3 style={{ color: '#4caf50', marginBottom: 16 }}><FaChartLine style={{ marginRight: 8 }} /> Price Chart & Technical Analysis</h3>
        <div style={{ marginBottom: 16, textAlign: 'right' }}>
          <label htmlFor="timeWindow" style={{ marginRight: 8, color: '#b0b8c1' }}>Time Window:</label>
          <select
            id="timeWindow"
            value={selectedTimeWindow}
            onChange={e => setSelectedTimeWindow(e.target.value)}
            style={{ padding: '6px 12px', borderRadius: 6, background: '#181c24', color: '#e1e5e9', border: '1px solid #23293a', fontSize: 15 }}
          >
            {timeWindows.map(win => (
              <option key={win.value} value={win.value}>{win.label}</option>
            ))}
          </select>
        </div>
        <div className="chart-container" style={{ minHeight: 180, background: '#181c24', borderRadius: 8, padding: 16, textAlign: 'center', color: '#b0b8c1' }}>
          {priceData && priceData.dates && priceData.close ? (
            <Plot
              data={[
                // Candlestick chart
                {
                  x: priceData.dates,
                  open: priceData.open,
                  high: priceData.high,
                  low: priceData.low,
                  close: priceData.close,
                  type: 'candlestick',
                  name: 'OHLC',
                },
                // SMA 20 overlay
                priceData.sma_20 && {
                  x: priceData.dates,
                  y: priceData.sma_20,
                  type: 'scatter',
                  mode: 'lines',
                  name: 'SMA 20',
                  line: { color: '#ff7f0e', width: 2 },
                },
                // SMA 50 overlay
                priceData.sma_50 && {
                  x: priceData.dates,
                  y: priceData.sma_50,
                  type: 'scatter',
                  mode: 'lines',
                  name: 'SMA 50',
                  line: { color: '#4caf50', width: 2, dash: 'dot' },
                },
                // Volume bar chart (secondary y-axis)
                priceData.volume && {
                  x: priceData.dates,
                  y: priceData.volume,
                  type: 'bar',
                  name: 'Volume',
                  marker: { color: '#b0b8c1' },
                  yaxis: 'y2',
                  opacity: 0.4,
                },
                // Daily returns line chart (third y-axis)
                priceData.daily_returns && {
                  x: priceData.dates,
                  y: priceData.daily_returns,
                  type: 'scatter',
                  mode: 'lines',
                  name: 'Daily Returns',
                  line: { color: '#e91e63', width: 1 },
                  yaxis: 'y3',
                },
              ].filter(Boolean)}
              layout={{
                title: `${currentSymbol || ''} Price Chart & Technicals`,
                paper_bgcolor: '#181c24',
                plot_bgcolor: '#181c24',
                font: { color: '#e1e5e9' },
                xaxis: { title: 'Date', color: '#e1e5e9', rangeslider: { visible: false } },
                yaxis: { title: 'Price (₹)', color: '#e1e5e9', domain: [0.3, 1] },
                yaxis2: { title: 'Volume', color: '#b0b8c1', overlaying: 'y', side: 'right', domain: [0, 0.2] },
                yaxis3: { title: 'Daily Returns', color: '#e91e63', overlaying: 'y', side: 'left', domain: [0, 0.1] },
                showlegend: true,
                margin: { t: 40, l: 40, r: 20, b: 40 },
                legend: { orientation: 'h', y: -0.2 },
                height: 400,
              }}
              style={{ width: '100%', height: '400px' }}
              config={{ responsive: true }}
            />
          ) : (
            <span style={{ fontSize: 18 }}>Select a symbol to view price chart & technical indicators</span>
          )}
        </div>
      </div>

      {/* Portfolio Comparison Tool - full width at bottom */}
      <div className="card" style={{ background: '#23293a', borderRadius: 16, boxShadow: '0 2px 12px #0003', padding: 24, margin: '0 5vw 32px 5vw' }}>
        <h3 style={{ color: '#4caf50', marginBottom: 16 }}><FaBalanceScale style={{ marginRight: 8 }} /> Portfolio Comparison Tool</h3>
        <div id="portfolioSection">
          <button className="btn" onClick={compareSelected} style={{ background: '#4caf50', color: '#fff', border: 'none', borderRadius: 8, padding: '10px 24px', fontSize: 18, fontWeight: 600, marginBottom: 16 }}>Compare Selected Assets</button>
          <div id="comparisonResults" style={{ marginTop: 20 }}>
            {portfolioMetrics && (
              <div>
                <h4 style={{ color: '#fff', marginBottom: 8 }}>Portfolio Comparison Results</h4>
                <div className="metrics" style={{ display: 'flex', gap: 18, marginBottom: 16 }}>
                  <div className="metric" style={{ background: '#181c24', borderRadius: 8, padding: 12, textAlign: 'center', flex: 1 }}>
                    <div className="metric-value" style={{ fontSize: 20, fontWeight: 700 }}>{portfolioMetrics.avg_return}%</div>
                    <div className="metric-label" style={{ color: '#b0b8c1' }}>Avg Return</div>
                  </div>
                  <div className="metric" style={{ background: '#181c24', borderRadius: 8, padding: 12, textAlign: 'center', flex: 1 }}>
                    <div className="metric-value" style={{ fontSize: 20, fontWeight: 700 }}>{portfolioMetrics.avg_volatility}%</div>
                    <div className="metric-label" style={{ color: '#b0b8c1' }}>Avg Volatility</div>
                  </div>
                  <div className="metric" style={{ background: '#181c24', borderRadius: 8, padding: 12, textAlign: 'center', flex: 1 }}>
                    <div className="metric-value" style={{ fontSize: 20, fontWeight: 700 }}>{portfolioMetrics.avg_score}</div>
                    <div className="metric-label" style={{ color: '#b0b8c1' }}>Avg Score</div>
                  </div>
                </div>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: 15, marginTop: 20 }}>
                  {comparisonResults && Object.entries(comparisonResults).map(([symbol, result]) => {
                    const score = result.score.final_score;
                    const scoreColor = score >= 75 ? '#4caf50' : score >= 50 ? '#ff9800' : '#f44336';
                    return (
                      <div key={symbol} style={{ border: '1px solid #23293a', padding: 15, borderRadius: 8, background: '#181c24', color: '#e1e5e9' }}>
                        <h5 style={{ color: '#4caf50', marginBottom: 4 }}>{symbol}</h5>
                        <p style={{ fontSize: '0.9em', color: '#b0b8c1' }}>{result.name}</p>
                        <p><strong>Score:</strong> <span style={{ color: scoreColor, fontWeight: 'bold' }}>{score}</span></p>
                        <p><strong>Return:</strong> {result.risk_metrics.annualized_return}%</p>
                        <p><strong>Risk:</strong> {result.risk_metrics.volatility}%</p>
                        <p><strong>Price:</strong> ₹{result.latest_price}</p>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default Dashboard;
