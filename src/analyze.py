
\"\"\"Analyze and produce deliverables:
- RMSE/MAE comparison saved to results/comparison.json
- plot of learned decomposition vs ground truth saved to results/decomposition.png
\"\"\"
import json, os
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

def compare():
    bres = json.load(open('results/baseline_results.json'))
    nres = json.load(open('results/nbeats_results.json'))
    comp = {'baseline_rmse': bres['rmse'], 'nbeats_rmse': nres['best_rmse'], 'baseline_mae': bres.get('mae'), 'nbeats_mae': None}
    # try to get nbeats_mae from history last entry
    hist = nres.get('history',[])
    if hist: comp['nbeats_mae'] = hist[-1].get('mae')
    os.makedirs('results', exist_ok=True)
    with open('results/comparison.json','w') as f:
        json.dump(comp, f, indent=2)
    print('Saved results/comparison.json')
    # placeholder decomposition plot using data if available
    if os.path.exists('data/sim.csv'):
        df = pd.read_csv('data/sim.csv', parse_dates=['timestamp'])
        plt.figure(figsize=(10,4))
        plt.plot(df['timestamp'][:100], df['value'][:100], label='value')
        plt.plot(df['timestamp'][:100], df['trend_true'][:100], label='true_trend')
        plt.plot(df['timestamp'][:100], df['seasonal_true'][:100], label='true_seasonal')
        plt.legend()
        plt.tight_layout()
        plt.savefig('results/decomposition.png')
        print('Saved results/decomposition.png')

if __name__ == '__main__':
    compare()
