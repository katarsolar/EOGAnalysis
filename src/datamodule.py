import os
from typing import Optional, Tuple, List
from pathlib import Path
import gdown

import h5py
import lightning as pl
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, random_split
import numpy as np

from src.dataset import EOGDataset
from src.configs import DataConfig
from src.logger import LOGGER
from src.constants import PROJECT_ROOT
logger = LOGGER

class EOGDataModule(pl.LightningDataModule):
    def __init__(self, data_cfg:DataConfig):
        super().__init__()
        self.cfg = data_cfg
        self.transforms = None

        self.train_dataset:Optional[EOGDataset] = None
        self.val_dataset:Optional[EOGDataset] = None
        self.test_dataset:Optional[EOGDataset] = None



    def setup(self, stage:str) -> None:
        if os.path.exists(self.cfg.data_path):
            logger.info('Data is already there.')
        else:
            logger.info("File is not found. Downloading h5py...")
            gdown.download(self.cfg.data_link, quiet=False)

        with h5py.File(self.cfg.data_path, 'r') as h5_file:
            signals = h5_file['windows'][()]
            labels = h5_file['labels'][()]
            logger.info('The data is opened.')
        if self.cfg.make_embeddings:
            pass
        logger.info(f'labels: {labels.shape}, signals: {signals.shape}')
        signals = np.transpose(signals, (2, 0, 1))# (4000, 4, 256)

        # basic transform.
        sig_mean = signals.mean(axis=(0,2), keepdims=True)
        sig_std = signals.std(axis=(0,2), keepdims=True)
        signals = (signals - sig_mean) / sig_std

        labels = np.transpose(labels, (1, 0))
        logger.info(f'labels after: {labels.shape}, signals after : {signals.shape}')
        X_train, X_temp, y_train, y_temp = train_test_split(signals, labels, test_size=0.25, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.65, random_state=42)
        logger.info("Dataset is splitted.")
        self.train_dataset = EOGDataset(X_train, y_train)
        self.val_dataset = EOGDataset(X_val, y_val)
        self.test_dataset = EOGDataset(X_test, y_test)








    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train_dataset,
                          batch_size=self.cfg.batch_size,
                          persistent_workers=True,
                          pin_memory=True,
                          num_workers=self.cfg.num_workers)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.val_dataset,
                          batch_size=self.cfg.batch_size,
                          persistent_workers=True,
                          pin_memory=True,
                          num_workers=self.cfg.num_workers)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test_dataset,
                          batch_size=self.cfg.batch_size,
                          persistent_workers=True,
                          pin_memory=True,
                          num_workers=self.cfg.num_workers)

