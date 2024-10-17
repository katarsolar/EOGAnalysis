class EOGDataModule(LightningDataModule):
    def __init__(self, data, window_size, step_size, batch_size=64):
        super().__init__()
        self.data = data
        self.window_size = window_size
        self.step_size = step_size
        self.batch_size = batch_size

    def setup(self, stage=None):
        dataset = EOGDataset(self.data, self.window_size, self.step_size)
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        self.train_dataset, self.test_dataset = random_split(dataset, [train_size, test_size])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)