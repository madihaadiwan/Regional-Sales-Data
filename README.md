# Regional Sales ML App

```bash
pip install -r requirements.txt
streamlit run app.py
```

The bundled `sales_data.csv` is your dataset (100,000 rows). Use **Upload file** in the sidebar to load an
Excel/CSV with the same columns (blank rows above the header are handled automatically).

| Tab | What it does |
|---|---|
| Data Analysis | KPIs, data-quality checks, descriptive stats, group-by, Actual-vs-Budget variance, correlations, pivot builder |
| Data Visualization | Trends, mix, seasonality heat-map, city map, leaderboards, pricing trends, custom chart builder |
| Predictive Analysis | Train Ridge / Random Forest / HistGradientBoosting to predict Sales, Profit or Units; diagnostics; permutation importance; what-if sale predictor |
| Forecasting | Monthly forecasts (Seasonal Naive, OLS trend+season, Holt-Winters, GBM-on-lags, Ensemble) with back-testing, intervals and segment breakdown |

Files: `app.py` (UI) · `core.py` (data/ML/forecast logic) · `sales_data.csv` (dataset)
"# Regional-Sales-Data" 
