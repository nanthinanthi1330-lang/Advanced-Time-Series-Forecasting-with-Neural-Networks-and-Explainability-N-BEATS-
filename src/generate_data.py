
"""Generate synthetic univariate time series with trend, multiple seasonalities and heteroscedastic noise.
Outputs CSV with columns: timestamp, value, trend_true, seasonal_true, noise_true"""
import argparse, math, csv, os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate(hours=24*365*3, hourly=True, seed=42):
    np.random.seed(seed)
    t = np.arange(hours)
    # trend: slow polynomial + occasional shift
    trend = 0.0001 * t + 0.0000001 * t**2 + 0.5 * (np.tanh((t-2000)/1000))
    # seasonalities: daily (24), weekly (168)
    daily = 2.0 * np.sin(2*np.pi * t / 24.0)
    weekly = 0.8 * np.sin(2*np.pi * t / 168.0 + 0.3)
    # multiple harmonics
    daily2 = 0.5 * np.sin(2*np.pi * t / 12.0)
    seasonal = daily + daily2 + weekly
    # heteroscedastic noise
    noise_scale = 0.5 + 0.001 * (t % 1000)
    noise = np.random.normal(scale=noise_scale)
    values = trend + seasonal + noise
    # timestamps
    start = datetime(2000,1,1)
    timestamps = [start + timedelta(hours=int(i)) for i in t]
    df = pd.DataFrame({'timestamp': timestamps, 'value': values, 'trend_true': trend, 'seasonal_true': seasonal, 'noise_true': noise})
    return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hours', type=int, default=24*365*3, help='number of hourly points (default ~3 years)')
    parser.add_argument('--out', type=str, default='data/sim.csv')
    args = parser.parse_args()
    df = generate(hours=args.hours)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    print('Saved', args.out)
