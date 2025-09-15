import torch, torch.nn as nn, torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import Subset
import random
from tqdm import tqdm

# Data transforms
transform = models.VGG11_Weights.IMAGENET1K_V1.transforms()

# Load dataset
train_dataset = ...
val_dataset = ...
test_dataset = ...

# Model (unchanged)
class VGG11(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # Model - using VGG11. This network is pretrained with the weights from
        # https://docs.pytorch.org/vision/main/models/generated/torchvision.models.vgg11.html
        backbone = torchvision.models.vgg11(weights='IMAGENET1K_V1')
        # Freeze all feature layers, we will only train the final classifier
        for param in model.features.parameters():
            param.requires_grad = False
        self.model.classifier[6] = nn.Linear(4096, 2)  # Change final layer for 2 classes
        self.fc2 = nn.Linear(32, 10)
        self.loss_fn = nn.CrossEntropyLoss()
        opt = optim.Adam() # We use a regular ADAM optimizer to adjust the model parameters

    def forward(self, x):
        x = self.pool(torch.relu(self.conv(x)))
        x = self.flat(x)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

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
        return optim.Adam(self.parameters())


# Training and evaluation
data_module = MNISTDataModule()
net = Vgg()
trainer = pl.Trainer(max_epochs=3, accelerator="auto", devices="auto")
trainer.fit(net, datamodule=data_module)
trainer.test(net, datamodule=data_module)
