

# Инициализация DataModule
data_module = EOGDataModule(
    data=EOG_data,
    window_size=WINDOW_SIZE,
    step_size=STEP_SIZE,
    batch_size=BATCH_SIZE
)

# Инициализация модели
model = DenoisingAutoencoderPL(num_channels=4, learning_rate=LEARNING_RATE)

# Инициализация Trainer
trainer = Trainer(
    max_epochs=EPOCHS,
    gpus=1 if torch.cuda.is_available() else 0,
    progress_bar_refresh_rate=20
)

# Обучение модели
trainer.fit(model, datamodule=data_module)
