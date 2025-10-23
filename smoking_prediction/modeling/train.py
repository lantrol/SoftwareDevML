import torch
import os
import pickle
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from smoking_prediction.modeling.model import VGG11
from smoking_prediction.data_loader import SmokerDataModule


def train_model(base_dir="../", batch_size=32, max_epochs=10, lr=1e-3):
    """
    Train a VGG11 model on the Smoker dataset using PyTorch Lightning.

    Parameters
    ----------
    data_dir : str, optional
        Base directory where dataset is located and checkpoints are saved
    batch_size : int, optional
        Batch size for training and validation (default is 32).
    max_epochs : int, optional
        Maximum number of training epochs (default is 10).
    lr : float, optional
        Learning rate for the Adam optimizer (default is 1e-3).

    Side Effects
    ------------
    - Saves the best model checkpoint in the "checkpoints" directory.
    - Saves training metrics (train/validation losses and validation accuracies) 
      as a pickle file in "reports/data/training_metrics.pkl".
    - Prints progress and information about training and testing.
    """
    # --- Check GPU availability ---
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    devices = 1 if torch.cuda.is_available() else "auto"
    print(f"Using accelerator: {accelerator}, devices: {devices or 'CPU'}")

     # --- Initialize data module and model ---
    data_module = SmokerDataModule(data_dir=base_dir + "/data", batch_size=batch_size, num_workers=0)
    net = VGG11(lr=lr)

    # --- Checkpoint callback ---
    checkpoint_callback = ModelCheckpoint(
        dirpath=base_dir + "/checkpoints",
        filename="vgg11-smoker-{epoch:02d}-{val_acc:.2f}",
        save_top_k=1,
        monitor="val_acc",
        mode="max",
    )

    # --- Trainer ---
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=devices,
        enable_progress_bar=True,
        log_every_n_steps=10,
        callbacks=[checkpoint_callback],
    )

    # --- Train and test ---
    print("Starting training...")
    trainer.fit(net, datamodule=data_module)
    print("Training finished.\nStarting testing...")
    trainer.test(net, datamodule=data_module)
    print("Testing finished.")

    print(f"Best checkpoint saved at: {checkpoint_callback.best_model_path}")
    
    # Save metrics to file for plotting
    metrics_data = {
        'train_losses': net.train_losses,
        'val_losses': net.val_losses,
        'val_accs': net.val_accs
    }

    # Derive metrics filename from checkpoint name
    ckpt_name = os.path.splitext(os.path.basename(checkpoint_callback.best_model_path))[0]
    save_dir = os.path.join(base_dir, "reports", "data")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{ckpt_name}_metrics.pkl")

    with open(save_path, "wb") as f:
        pickle.dump(metrics_data, f)

    print(f"Training metrics saved to '{save_path}'")

    return checkpoint_callback.best_model_path, save_path

if __name__ == "__main__":
    train_model()
    #print('All imports done right')
    pass
