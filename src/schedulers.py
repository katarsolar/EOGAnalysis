import os

import lightning
from lightning import Trainer
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger
from lightning.pytorch.callbacks import ModelSummary, ModelCheckpoint, LearningRateMonitor, ProgressBar, EarlyStopping


from src.logger import LOGGER
from src.configs import ExpConfig, ModelConfig, DataConfig, DatasetConfig
from src.lt_module import SimSiamModule
from src.datamodule import ClothingDataModule



def train(cfg:ExpConfig):
    lightning.seed_everything(0)
    # TODO: изменить filename.
    callbacks = [
        ModelCheckpoint(monitor='train_loss', save_top_k=1, mode='min', filename='best-checkpoint'),
        ModelSummary(),
        LearningRateMonitor(logging_interval='epoch'),
        ProgressBar(),
    ]
    # if cfg['track_in_clearml']:
    #     pass
    model = SimSiamModule(config=ModelConfig())
    datamodule = ClothingDataModule(cfg=DataConfig())

    logger = TensorBoardLogger(save_dir='src/logs', name='simsiam_experiment')
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
