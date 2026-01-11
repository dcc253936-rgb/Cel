import torch
import torch.nn as nn

class BFUpdate(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 32),
            nn.ReLU(),
            nn.Linear(32, 3)
        )

    def forward(self, pred, meas):
        # pred: (B, 3) 预测位置
        # meas: (B, 3) 当前测量
        x = torch.cat([pred, meas], dim=-1)
        return self.net(x)
