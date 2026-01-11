import torch
import torch.nn as nn
from models.causal_cnn import CausalCNN
from models.attention import SelfAttention

class CALSTM(nn.Module):
    def __init__(self):
        super().__init__()

        self.causal = CausalCNN(6, 32)
        self.lstm = nn.LSTM(
            input_size=6 + 32,
            hidden_size=32,
            batch_first=True
        )
        self.attn = SelfAttention(32)
        self.fc = nn.Linear(32, 3)

    def forward(self, x):
        causal_feat = self.causal(x)
        x = torch.cat([x, causal_feat], dim=-1)

        lstm_out, _ = self.lstm(x)
        attn_out = self.attn(lstm_out)

        return self.fc(attn_out)
