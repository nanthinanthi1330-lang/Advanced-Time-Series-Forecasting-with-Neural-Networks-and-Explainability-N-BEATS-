
import pandas as pd
import numpy as np
from torch.utils.data import Dataset

class TimeSeriesDataset(Dataset):
    def __init__(self, series, input_len=168, horizon=24):
        # series: 1D numpy array or pandas Series
        self.series = np.asarray(series).astype('float32')
        self.input_len = input_len
        self.horizon = horizon
        self.indices = list(range(0, len(self.series) - input_len - horizon + 1))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        x = self.series[i:i+self.input_len]
        y = self.series[i+self.input_len:i+self.input_len+self.horizon]
        return x, y
