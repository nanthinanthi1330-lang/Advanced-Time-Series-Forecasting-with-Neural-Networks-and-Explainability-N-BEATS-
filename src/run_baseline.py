
\"\"\"Baseline using STL decomposition (seasonal + trend) + ARIMA on residuals.
Requires statsmodels. Saves baseline_results.json with RMSE/MAE.\"\"\"
import json, os
import numpy as np, pandas as pd
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error

def run_baseline(csv_path, input_len=168, horizon=24):
    df = pd.read_csv(csv_path, parse_dates=['timestamp'])
    series = df['value'].values
    n_train = int(len(series)*0.8)
    train = series[:n_train]
    test = series[n_train:]
    # apply STL on full train to extract seasonal and trend
    stl = STL(train, period=24)
    res = stl.fit()
    trend = res.trend
    seasonal = res.seasonal
    resid = train - trend - seasonal
    # fit a simple ARIMA on residuals
    model = ARIMA(resid, order=(1,0,1)).fit()
    # forecast residuals for test horizon in rolling fashion (simple approach)
    preds = []
    for i in range(0, len(test), horizon):
        # here we forecast next horizon residuals (naive using last model)
        f = model.predict(start=len(resid), end=len(resid)+horizon-1)
        # reconstruct using last trend and seasonal slice (approx)
        last_trend = trend[-horizon:]
        last_seas = seasonal[-horizon:]
        pred = f + last_trend + last_seas
        preds.extend(pred)
    preds = np.array(preds)[:len(test)]
    rmse = mean_squared_error(test[:len(preds)], preds, squared=False)
    mae = mean_absolute_error(test[:len(preds)], preds)
    os.makedirs('results', exist_ok=True)
    with open('results/baseline_results.json','w') as f:
        json.dump({'rmse': float(rmse), 'mae': float(mae)}, f, indent=2)
    print('Saved results/baseline_results.json')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, default='data/sim.csv')
    args = parser.parse_args()
    run_baseline(args.csv)
