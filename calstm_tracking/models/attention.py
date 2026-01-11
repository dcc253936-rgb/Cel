import torch
import torch.nn as nn
import math

class SelfAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)

    def forward(self, x):
        # x: (B, T, D)
        Q = self.q(x)
        K = self.k(x)
        V = self.v(x)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(Q.size(-1))
        weights = torch.softmax(scores, dim=-1)

        out = torch.matmul(weights, V)
        return out.mean(dim=1)
