import os
from typing import Dict, Any

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn
import timm

from src.logger import LOGGER
from src.schedulers import get_cosine_schedule_with_warmup
from src.configs import ModelConfig
from lightning.pytorch import LightningModule

logger = LOGGER


class CNN1DModel(nn.Module):
    def __init__(self, num_classes=2):
        super(CNN1DModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=64, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(64)

        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(128)

        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc2(F.relu(self.fc1(x))))
        x = self.fc3(x)
        return x


class EOGLightningModule(LightningModule):
    def __init__(self, config: ModelConfig):
        super(EOGLightningModule, self).__init__()
        self.config = config
        self.skeleton_model = CNN1DModel(num_classes=2)
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = x.float()
        return self.skeleton_model(x)

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        inputs = inputs.float()
        targets = targets.float()
        outputs = self.forward(inputs)
        loss = self.loss_fn(outputs, targets)


        rmse = torch.sqrt(loss)

        # Log both training loss and RMSE
        self.log("train_loss", loss, prog_bar=True, logger=True)
        self.log("train_rmse", rmse, prog_bar=True, logger=True)
        logger.info(f'Training Loss: {loss.item()}, Training RMSE: {rmse.item()}')

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self.forward(inputs)
        val_loss = self.loss_fn(outputs, targets)

        # Calculate RMSE
        rmse = torch.sqrt(val_loss)

        # Log both validation loss and RMSE
        self.log("val_loss", val_loss, prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val_rmse", rmse, prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        logger.info(f'Validation Loss: {val_loss.item()}, Validation RMSE: {rmse.item()}')

        return {"val_loss": val_loss, "val_rmse": rmse}

    def test_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self.forward(inputs)
        test_loss = self.loss_fn(outputs, targets)
        self.log("test_loss", test_loss, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        logger.info(f'Test Loss: {test_loss.item()}')
        return test_loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.config.learning_rate)
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=10, num_training_steps=100)
        return [optimizer], [scheduler]
