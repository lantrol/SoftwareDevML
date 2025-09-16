import os
import pytorch_lightning as pl
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader


class SmokerDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str =  "../data", batch_size = 32, num_workers=8):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = models.VGG11_Weights.IMAGENET1K_V1.transforms()


    def setup(self, stage=None):
        # Expecting folders: data/train/smoker, data/train/non-smoker, etc.
        self.train_dataset = datasets.ImageFolder(root=f"{self.data_dir}/train", transform=self.transform)
        self.val_dataset = datasets.ImageFolder(root=f"{self.data_dir}/val", transform=self.transform)
        self.test_dataset = datasets.ImageFolder(root=f"{self.data_dir}/test", transform=self.transform)


    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)


    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)


    def test_dataloader(self):
     return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)