import torch, torch.nn as nn, torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import Subset
import random
from tqdm import tqdm
from src.data_loader import SmokerDataModule
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import pickle
import os


# Model with enhanced logging
class VGG11(pl.LightningModule):
    def __init__(self, lr=1e-3):
        super().__init__()
        num_classes = 2
        # Model - using VGG11. This network is pretrained with the weights from
        # https://docs.pytorch.org/vision/main/models/generated/torchvision.models.vgg11.html
        backbone = models.vgg11(weights='IMAGENET1K_V1')
        # Freeze all feature layers, we will only train the final classifier
        # Extract features (remove classifier head)
        self.feature_extractor = backbone.features
        for param in self.feature_extractor.parameters():
            param.requires_grad = False  # freeze

        # Get number of features from last pooling layer
        num_filters = backbone.classifier[0].in_features  # = 25088 for VGG11

        # Define classifier for binary classification
        self.classifier = nn.Linear(num_filters, num_classes)

        self.loss_fn = nn.CrossEntropyLoss()
        self.lr = lr
        
        # Lists to store metrics for plotting
        self.train_losses = []
        self.val_losses = []
        self.val_accs = []

    def forward(self, x):
        feats = self.feature_extractor(x)
        feats = torch.flatten(feats, 1)
        out = self.classifier(feats)
        return out

    def training_step(self, batch, batch_idx):
        xb, yb = batch
        out = self(xb)
        loss = self.loss_fn(out, yb)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        xb, yb = batch
        out = self(xb)
        loss = self.loss_fn(out, yb)
        preds = out.argmax(1)
        acc = (preds == yb).float().mean()
        
        # Log both loss and accuracy
        self.log('val_loss', loss, prog_bar=True, on_epoch=True)
        self.log('val_acc', acc, prog_bar=True, on_epoch=True)
        
        return {"val_loss": loss, "val_acc": acc}

    def on_train_epoch_end(self):
        # Get the logged training loss for this epoch
        train_loss = self.trainer.callback_metrics.get('train_loss')
        if train_loss is not None:
            self.train_losses.append(train_loss.item())

    def on_validation_epoch_end(self):
        # Get the logged validation metrics for this epoch
        val_loss = self.trainer.callback_metrics.get('val_loss')
        val_acc = self.trainer.callback_metrics.get('val_acc')
        
        if val_loss is not None:
            self.val_losses.append(val_loss.item())
        if val_acc is not None:
            self.val_accs.append(val_acc.item())

    def test_step(self, batch, batch_idx):
        xb, yb = batch
        out = self(xb)
        preds = out.argmax(1)
        acc = (preds == yb).float().mean()
        self.log('test_acc', acc, prog_bar=True, on_epoch=True)

    def configure_optimizers(self):
        return optim.Adam(self.classifier.parameters(), lr=self.lr)