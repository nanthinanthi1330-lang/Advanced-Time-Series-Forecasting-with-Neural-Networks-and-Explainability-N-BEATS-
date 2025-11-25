
\"\"\"Train script for simplified N-BEATS model. Saves best model and results JSON with RMSE/MAE.\"\"\"
import argparse, json, os, time
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from nbeats import NBeats
from dataset import TimeSeriesDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error

def prepare_data(csv_path, input_len, horizon, split=0.8):
    df = pd.read_csv(csv_path, parse_dates=['timestamp'])
    series = df['value'].values
    n_train = int(len(series)*split)
    train = series[:n_train]
    val = series[n_train - input_len - horizon + 1:]
    return train, val

def train_loop(model, loader, loss_fn, opt, device):
    model.train()
    total=0.0; count=0
    for x,y in loader:
        x = x.to(device); y = y.to(device)
        opt.zero_grad()
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()
        opt.step()
        total += loss.item()*x.size(0)
        count += x.size(0)
    return total/count

def evaluate(model, loader, device):
    model.eval()
    preds=[]; trues=[]
    with torch.no_grad():
        for x,y in loader:
            x = x.to(device)
            out = model(x)
            preds.append(out.cpu().numpy())
            trues.append(y.numpy())
    if not preds: return None
    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    rmse = mean_squared_error(trues.flatten(), preds.flatten(), squared=False)
    mae = mean_absolute_error(trues.flatten(), preds.flatten())
    return rmse, mae

def run(cfg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_s, val_s = prepare_data(cfg['data_csv'], cfg['input_len'], cfg['horizon'], split=cfg.get('split',0.8))
    train_ds = TimeSeriesDataset(train_s, cfg['input_len'], cfg['horizon'])
    val_ds = TimeSeriesDataset(val_s, cfg['input_len'], cfg['horizon'])
    train_loader = DataLoader(train_ds, batch_size=cfg['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg['batch_size'], shuffle=False)
    model = NBeats(cfg['input_len'], cfg['horizon'], hidden_size=cfg['hidden_size'], n_blocks=cfg['n_blocks'], n_layers=cfg['n_layers']).to(device)
    loss_fn = nn.MSELoss()
    opt = optim.Adam(model.parameters(), lr=cfg['lr'])
    best_rmse = float('inf'); best_state=None; history = []
    for e in range(cfg['epochs']):
        tr_loss = train_loop(model, train_loader, loss_fn, opt, device)
        vals = evaluate(model, val_loader, device)
        if vals is None: break
        rmse, mae = vals
        history.append({'epoch':e+1,'rmse':rmse,'mae':mae})
        if rmse < best_rmse:
            best_rmse = rmse
            best_state = model.state_dict()
    os.makedirs('results', exist_ok=True)
    if best_state:
        torch.save(best_state, 'results/best_nbeats.pth')
    with open('results/nbeats_results.json','w') as f:
        json.dump({'best_rmse': best_rmse, 'history': history, 'config': cfg}, f, indent=2)
    print('Saved results/nbeats_results.json')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/nbeats_config.json')
    args = parser.parse_args()
    cfg = json.load(open(args.config))
    run(cfg)
