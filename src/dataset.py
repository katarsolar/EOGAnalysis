from torch.utils.data import Dataset
from src.logger import LOGGER
import torch
from src.constants import PROJECT_ROOT
logger = LOGGER


class EOGDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        super(EOGDataset, self).__init__()
        self.transform = transform
        self.data = data
        self.labels = labels



    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        signal = self.data[idx,:,:]
        signal = torch.transpose(torch.Tensor(signal), 1, 0)
        label = self.labels[idx, :]
        if self.transform:
            signal = self.transform(signal)
        return signal, label





