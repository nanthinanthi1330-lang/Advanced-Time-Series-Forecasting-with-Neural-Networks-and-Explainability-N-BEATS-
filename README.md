
# Advanced Time Series Forecasting with N-BEATS (Assignment-ready)

This project implements:
- programmatic synthetic time-series generator (trend, multiple seasonality, heteroscedastic noise)
- simplified N-BEATS model in PyTorch for multi-step forecasting
- baseline pipeline using STL + ARIMA (statsmodels)
- training, evaluation (MAE, RMSE), and decomposition-based explainability outputs
- analysis scripts and a short report matching deliverables

Quick start:
1. Create venv and install requirements: `pip install -r requirements.txt`
2. Generate data: `python src/generate_data.py --hours 24 --years 3 --out data/sim.csv`
3. Train N-BEATS: `python src/train_nbeats.py --config configs/nbeats_config.json`
4. Run baseline: `python src/run_baseline.py`
5. Analyze and visualize: `python src/analyze.py`

See `report/REPORT.md` for experiment protocol and required deliverables.
