import os
from datetime import datetime
import lightning
from lightning import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelSummary, ModelCheckpoint, LearningRateMonitor, ProgressBar

from src.logger import LOGGER
from src.configs import ExpConfig, ModelConfig, DataConfig
from src.lt_module import EOGLightningModule
from src.datamodule import EOGDataModule


def train(cfg: ExpConfig):
    lightning.seed_everything(0)

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        save_top_k=1,
        mode='min',
        dirpath='logs/checkpoints',
        filename=f'best_model_{datetime.now().strftime("%Y%m%d-%H%M%S")}',
        every_n_epochs=1
    )

    logger = TensorBoardLogger(
        save_dir=os.path.join('logs', 'boards', datetime.now().strftime("%Y%m%d-%H%M%S")),
        name='base_experiment1'
    )

    callbacks = [
        checkpoint_callback,
        ModelSummary(),
        LearningRateMonitor(logging_interval='epoch'),
        ProgressBar()
    ]

    model = EOGLightningModule(config=ModelConfig())
    datamodule = EOGDataModule(data_cfg=DataConfig())

    trainer = Trainer(
        **dict(cfg.trainer_config),
        callbacks=callbacks,
        logger=logger,
        max_epochs=100,
        log_every_n_steps=100
    )

    trainer.fit(
        model=model,
        datamodule=datamodule,
    )


if __name__ == '__main__':
    train(cfg=ExpConfig())

