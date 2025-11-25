
\"\"\"Minimal N-BEATS-like implementation for univariate forecasting.
This is simplified for assignment/demo purposes (not full production N-BEATS).
\"\"\"
import torch
import torch.nn as nn

class NBeatsBlock(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers, theta_size):
        super().__init__()
        layers = []
        for _ in range(n_layers):
            layers.append(nn.Linear(input_size if _==0 else hidden_size, hidden_size))
            layers.append(nn.ReLU())
        self.fc = nn.Sequential(*layers)
        self.theta = nn.Linear(hidden_size, theta_size)

    def forward(self, x):
        x = self.fc(x)
        return self.theta(x)

class NBeats(nn.Module):
    def __init__(self, input_len, horizon, hidden_size=128, n_blocks=3, n_layers=2):
        super().__init__()
        self.input_len = input_len
        self.horizon = horizon
        self.blocks = nn.ModuleList()
        for _ in range(n_blocks):
            # theta size = input_len + horizon (trend + seasonality style)
            theta_size = input_len + horizon
            self.blocks.append(NBeatsBlock(input_len, hidden_size, n_layers, theta_size))
        # final linear to map aggregated thetas to horizon
        self.final = nn.Linear((input_len + horizon)*n_blocks, horizon)

    def forward(self, x):
        # x shape: (batch, input_len)
        thetas = []
        for b in self.blocks:
            theta = b(x)  # (batch, theta_size)
            thetas.append(theta)
        thetas = torch.cat(thetas, dim=1)
        out = self.final(thetas)
        return out
