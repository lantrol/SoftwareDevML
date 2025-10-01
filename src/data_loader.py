import os
import pytorch_lightning as pl
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader


class SmokerDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for the Smoker dataset.

    Handles training, validation, and test datasets for binary classification
    (smoker vs. non-smoker) with VGG11 preprocessing.

    Parameters
    ----------
    data_dir : str, optional
        Root directory containing 'train', 'val', and 'test' subfolders 
        (default "../data").
    batch_size : int, optional
        Batch size for the DataLoaders (default 32).
    num_workers : int, optional
        Number of subprocesses to use for data loading (default 8).

    Attributes
    ----------
    transform : torchvision.transforms
        Preprocessing transforms for VGG11 pretrained weights.
    train_dataset : torchvision.datasets.ImageFolder
        Training dataset.
    val_dataset : torchvision.datasets.ImageFolder
        Validation dataset.
    test_dataset : torchvision.datasets.ImageFolder
        Test dataset.
    """
    def __init__(self, data_dir: str =  "../data", batch_size = 32, num_workers=8):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = models.VGG11_Weights.IMAGENET1K_V1.transforms()


    def setup(self, stage=None):
        """
        Set up datasets for training, validation, and testing.

        Parameters
        ----------
        stage : str, optional
            Stage of setup ("fit", "test", or None). Not used here.
        """
        # Expecting folders: data/train/smoker, data/train/non-smoker, etc.
        self.train_dataset = datasets.ImageFolder(root=f"{self.data_dir}/train", transform=self.transform)
        self.val_dataset = datasets.ImageFolder(root=f"{self.data_dir}/val", transform=self.transform)
        self.test_dataset = datasets.ImageFolder(root=f"{self.data_dir}/test", transform=self.transform)


    def train_dataloader(self):
        """
        Returns the DataLoader for the training dataset.

        Returns
        -------
        torch.utils.data.DataLoader
            DataLoader yielding batches from the training dataset.
        """
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)


    def val_dataloader(self):
        """
        Returns the DataLoader for the validation dataset.

        Returns
        -------
        torch.utils.data.DataLoader
            DataLoader yielding batches from the validation dataset.
        """
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)


    def test_dataloader(self):
        """
        Returns the DataLoader for the test dataset.

        Returns
        -------
        torch.utils.data.DataLoader
            DataLoader yielding batches from the test dataset.
        """
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)