import os

import lightning
from lightning import Trainer
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger
from lightning.pytorch.callbacks import ModelSummary, ModelCheckpoint, LearningRateMonitor, ProgressBar, EarlyStopping


from src.logger import LOGGER
from src.configs import ExpConfig, ModelConfig, DataConfig, DatasetConfig
from src.lt_module import EOGLightningModule
from src.datamodule import EOGDataModule



def train(cfg:ExpConfig):
    lightning.seed_everything(0)
    callbacks = [
        ModelCheckpoint(monitor='val_loss', save_top_k=1, mode='min', filename='best-checkpoint'),
        ModelSummary(),
        LearningRateMonitor(logging_interval='epoch'),
        ProgressBar(),
    ]
    # if cfg['track_in_clearml']:
    #     pass
    model = EOGLightningModule(config=ModelConfig())
    datamodule = EOGDataModule(data_cfg=DataConfig())

    logger = TensorBoardLogger(save_dir='/logs', name='base_experiment1')
    # csv_logger = CSVLogger(save_dir='logs/', name='simsiam_experiment_csv')


    trainer = Trainer(
        **dict(cfg.trainer_config),
        callbacks=callbacks,
        logger=[logger]
    )

    trainer.fit(
        model=model,
        datamodule=datamodule)




if __name__ == '__main__':
    train(cfg=ExpConfig())
