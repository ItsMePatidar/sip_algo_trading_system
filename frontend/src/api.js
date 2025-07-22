const base_url = 'https://sip-algo-trading-system.onrender.com';

export const apiSearchSymbol = async (query) => {
  const res = await fetch(`${base_url}/api/symbols/search?q=${query}&limit=10`);
  return res.json();
};
export const apiFetchPriceData = async (symbol, period = '1d') => {
  const res = await fetch(`${base_url}/api/price_data/${symbol}?period=${period}`);
  return res.json();
};
export const apiFetchMLPipelineResult = async (symbol) => {
  const res = await fetch(`${base_url}/api/ml_pipeline_result?symbol=${symbol}`);
  return res.json();
}
export const apiFetchAnalysis = async (symbol, forecastDays = 7) => {
  const res = await fetch(`${base_url}/api/analysis/${symbol}?forecast_days=${forecastDays}`);
  return res.json();
}

export const apiFetchCategories = async () => {
  return 1;
}
