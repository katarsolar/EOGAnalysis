from torch.utils.data import Dataset
import numpy as np
import torch


class EOGDataset(Dataset):
    def __init__(self, data, window_size, step_size):

        self.data = data
        self.window_size = window_size
        self.step_size = step_size
        self.windows = self.create_windows()

    def create_windows(self):
        X = []
        y = []
        num_channels, num_samples = self.data.shape
        for start in range(0, num_samples - self.window_size, self.step_size):
            end = start + self.window_size
            window = self.data[:, start:end]
            X.append(window)
            y.append(window)
        return np.array(X), np.array(y)

    def __len__(self):
        return self.windows[0].shape[0]

    def __getitem__(self, idx):
        noisy = self.windows[0][idx]
        clean = self.windows[1][idx]

        return torch.tensor(noisy, dtype=torch.float32), torch.tensor(clean, dtype=torch.float32)