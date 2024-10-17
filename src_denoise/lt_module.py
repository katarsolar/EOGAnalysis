class DenoisingAutoencoderPL(LightningModule):
    def __init__(self, num_channels=4, learning_rate=1e-3):
        super(DenoisingAutoencoderPL, self).__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate


        self.encoder = nn.Sequential(
            nn.Conv1d(num_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2, 2),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2, 2)
        )

        # Декодер
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(32, 16, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose1d(16, num_channels, kernel_size=2, stride=2),
            nn.ReLU()
        )

        self.criterion = nn.MSELoss()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def training_step(self, batch, batch_idx):
        noisy, clean = batch
        outputs = self.forward(noisy)
        loss = self.criterion(outputs, clean)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        noisy, clean = batch
        outputs = self.forward(noisy)
        loss = self.criterion(outputs, clean)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
