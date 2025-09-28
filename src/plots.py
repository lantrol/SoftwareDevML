import torch
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pytorch_lightning as pl
import os

from model import VGG11
from data_loader import SmokerDataModule

# --- Device ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load checkpoint ---
BASE_DIR = Path(__file__).parent          # -> SoftwareDevML/src
CHECKPOINT_DIR = BASE_DIR.parent / "checkpoints"  # -> SoftwareDevML/checkpoints
ckpt_path = CHECKPOINT_DIR / "vgg11-smoker-epoch=02-val_acc=0.88.ckpt"
print("Resolved checkpoint path:", ckpt_path)
model = VGG11.load_from_checkpoint(ckpt_path)
model = model.to(device)
model.eval()

# --- Load test data ---
data_module = SmokerDataModule(data_dir="../data", batch_size=32, num_workers=0)
data_module.setup()
test_loader = data_module.test_dataloader()
class_names = data_module.train_dataset.classes

# --- Collect predictions & labels ---
all_preds = []
all_labels = []

with torch.no_grad():
    for xb, yb in test_loader:
        xb, yb = xb.to(device), yb.to(device)   # move data to same device as model
        logits = model(xb)
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(yb.cpu().numpy())

# --- Confusion Matrix ---
cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

fig, ax = plt.subplots(figsize=(6, 6))
disp.plot(cmap="Blues", ax=ax, values_format="d", colorbar=False)
plt.title("Confusion Matrix (Test Set)")
plt.show()
