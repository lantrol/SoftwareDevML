import torch, torch.nn as nn, torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import Subset
import random
from tqdm import tqdm
from data_loader import SmokerDataModule
import pytorch_lightning as pl


# Model (unchanged)
class VGG11(pl.LightningModule):
    def __init__(self, lr = 1e-3):
        super().__init__()
        num_classes = 2
        # Model - using VGG11. This network is pretrained with the weights from
        # https://docs.pytorch.org/vision/main/models/generated/torchvision.models.vgg11.html
        backbone = models.vgg11(weights='IMAGENET1K_V1')
        # Freeze all feature layers, we will only train the final classifier
        # Extract features (remove classifier head)
        self.feature_extractor = backbone.features
        for param in self.feature_extractor.parameters():
            param.requires_grad = False  # freezeç

        # Get number of features from last pooling layer
        num_filters = backbone.classifier[0].in_features  # = 25088 for VGG11

        # Define classifier for binary classification
        self.classifier = nn.Linear(num_filters, num_classes)

        self.loss_fn = nn.CrossEntropyLoss()
        self.lr = lr

    def forward(self, x):
        with torch.no_grad():
            feats = self.feature_extractor(x)
            feats = torch.flatten(feats, 1)
        out = self.classifier(feats)
        return out

    def training_step(self, batch, batch_idx):
        xb, yb = batch
        out = self(xb)
        loss = self.loss_fn(out, yb)
        return loss

    def validation_step(self, batch, batch_idx):
        xb, yb = batch
        out = self(xb)
        preds = out.argmax(1)
        acc = (preds == yb).float().mean()
        self.log('val_acc', acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        xb, yb = batch
        out = self(xb)
        preds = out.argmax(1)
        acc = (preds == yb).float().mean()
        self.log('test_acc', acc, prog_bar=True, on_epoch=True)

    def configure_optimizers(self):
        return optim.Adam(self.classifier.parameters(), lr = self.lr)


if __name__ == "__main__":
    data_module = SmokerDataModule(data_dir="../data", batch_size=32) 
    net = VGG11()

    # Trainer
    trainer = pl.Trainer(max_epochs=3, accelerator="auto", devices="auto", )

    # Train/test
    trainer.fit(net, datamodule=data_module)
    trainer.test(net, datamodule=data_module)
