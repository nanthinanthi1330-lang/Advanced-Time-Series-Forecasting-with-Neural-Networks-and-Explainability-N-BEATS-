
# Report: N-BEATS Time Series Forecasting (Assignment Deliverables)

## 1. Data generation
- Generated hourly data (~3 years) with trend (slow polynomial + shift), daily and weekly seasonality, and heteroscedastic noise.
- Ground-truth components saved in `data/sim.csv` as trend_true, seasonal_true, noise_true.

## 2. Model
- Simplified N-BEATS implementation in `src/nbeats.py` with block structure. Designed for explainability by comparing learned outputs to ground truth components.

## 3. Baseline
- STL decomposition + ARIMA on residuals (`src/run_baseline.py`).

## 4. Protocol
- Train/validation split: 80% train, 20% test. Use fixed budgets (example config uses 5 epochs for speed; assignment recommends 30-50 trials/epochs for full runs).
- Evaluation metrics: RMSE and MAE. Save results to `results/` and produce `results/comparison.json` and `results/decomposition.png`.

## 5. Deliverables included
1. Runnable code for data generation, model training, baseline and analysis (`src/`). 
2. Text-based analysis and short report (`report/REPORT.md`). 
3. Decomposition visuals and comparison summary (`results/` after running scripts).
4. Config example in `configs/nbeats_config.json`.

