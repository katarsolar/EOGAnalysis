import os
from typing import Dict, Any


import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn
import timm

from src.logger import LOGGER
from src.schedulers import get_cosine_schedule_with_warmup
# from src.transforms import get_augmentations
from src.configs import ModelConfig
from lightning.pytorch import LightningModule


class MLPHead(nn.Module):
    def __init__(self, in_dim, hidden_dim, mid_dim, out_dim):
        super(MLPHead, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, mid_dim)
        self.bn2 = nn.BatchNorm1d(mid_dim)
        self.fc3 = nn.Linear(mid_dim, out_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.fc3(x)
        return x


class CNN1DModel(nn.Module):
    def __init__(self, num_classes):
        super(CNN1DModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=16, kernel_size=3, stride=1, padding=1)
        pin_memory=True,
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 256, num_classes)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(in_features=32 * 256, out_features=4096)
        self.pool = nn.MaxPool1d(2)
        self.fc3 = nn.Linear(4096, out_features=num_classes)


    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # развертываем перед полносвязным слоем
        x = self.fc(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x



class EOGLightningModule(LightningModule):
    def __init__(self, config: ModelConfig):
        super(EOGLightningModule, self).__init__()
        self.config = config
        self.skeleton_model = CNN1DModel(num_classes=2)
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        x = x.float()
        return self.skeleton_model(x)

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self.forward(inputs)
        loss = self.loss_fn(outputs, targets)
        self.log("train_loss", loss, prog_bar=True, logger=True, on_step=True)
        return {"train_loss": loss}

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self.forward(inputs)
        val_loss = self.loss_fn(outputs, targets)
        self.log("val_loss", val_loss, prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        return {"val_loss": val_loss}

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.config.learning_rate)
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=10, num_training_steps=100)
        return [optimizer], [scheduler]
