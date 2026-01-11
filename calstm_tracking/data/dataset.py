"""
dataset.py
----------
将轨迹数据切分为可用于训练 CALSTM 的 Dataset

输入：
    trajectory.npz
        - state_true   (N, 6)
        - measurement  (N, 6)

输出：
    X: (num_samples, T, 6)
    y: (num_samples, 3)
"""

import numpy as np
import torch
from torch.utils.data import Dataset


class TrackingDataset(Dataset):
    def __init__(
        self,
        npz_path,
        window_size=10,
        stride=1,
        use_measurement=True
    ):
        """
        参数说明：
        ----------
        npz_path : str
            trajectory.npz 路径
        window_size : int
            时间窗长度 T
        stride : int
            滑动步长
        use_measurement : bool
            True  -> 用 measurement 作为输入
            False -> 用 state_true 作为输入（调试用）
        """
        data = np.load(npz_path)

        self.state_true = data["state_true"]
        self.measurement = data["measurement"]

        self.window_size = window_size
        self.stride = stride
        self.use_measurement = use_measurement

        self.inputs, self.targets = self._build_samples()

    def _build_samples(self):
        inputs = []
        targets = []

        total_len = len(self.state_true)

        for start in range(
            0,
            total_len - self.window_size - 1,
            self.stride
        ):
            end = start + self.window_size

            if self.use_measurement:
                x = self.measurement[start:end]
            else:
                x = self.state_true[start:end]

            # 预测目标：下一时刻真实位置 (x, y, z)
            y = self.state_true[end, 0:3]

            inputs.append(x)
            targets.append(y)

        inputs = np.array(inputs, dtype=np.float32)
        targets = np.array(targets, dtype=np.float32)

        return inputs, targets

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.inputs[idx])
        y = torch.from_numpy(self.targets[idx])
        return x, y
