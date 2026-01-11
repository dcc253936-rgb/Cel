import torch
import torch.nn as nn

class CausalCNN(nn.Module):
    def __init__(self, in_channels=6, hidden=32, kernel_size=3):
        super().__init__()

        self.kernel_size = kernel_size

        self.conv1 = nn.Conv1d(
            in_channels,
            hidden,
            kernel_size=kernel_size,
            padding=kernel_size - 1
        )

        self.conv2 = nn.Conv1d(
            hidden,
            hidden,
            kernel_size=kernel_size,
            padding=kernel_size - 1
        )

        self.relu = nn.ReLU()

    def forward(self, x):
        """
        x: (B, T, C)
        return: (B, T, hidden)
        """
        T = x.size(1)

        x = x.transpose(1, 2)   # (B, C, T)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))

        # 🔴 核心修复：裁剪到原始时间长度
        x = x[:, :, :T]

        x = x.transpose(1, 2)   # (B, T, hidden)
        return x
