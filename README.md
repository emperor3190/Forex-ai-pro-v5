# Forex AI Pro V9.1

Integrated V9.1 Streamlit research application.

## Included
- Twelve Data market data
- Data-quality/freshness/gap gates
- Technical indicators
- Market structure, liquidity sweep and FVG detection
- Multi-timeframe confluence
- Logistic Regression and Random Forest probability models
- Chronological validation and calibration
- Forex decision engine
- Binary-options research engine with payout/EV/break-even gates
- Strict walk-forward backtesting
- SQLite signal journal
- Plotly market chart

## Run
1. Install requirements:
   `pip install -r requirements.txt`
2. Configure `TWELVE_DATA_API_KEY` in Streamlit secrets or environment.
3. Start:
   `streamlit run app.py`

This is a research/paper-trading application. It does not guarantee profits.
Broker binary strike/settlement quotes may differ from Twelve Data.
