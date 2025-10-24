import torch
from smoking_prediction.modeling.model import VGG11
from smoking_prediction.data_loader import SmokerDataModule

def load_model(checkpoint_path, lr=1e-3):
    """
    Load a pretrained VGG11 model from a checkpoint.

    Parameters
    ----------
    checkpoint_path : str
        Path to the checkpoint file (.ckpt).
    lr : float, optional
        Learning rate for the optimizer (default is 1e-3). Only used for loading the LightningModule.

    Returns
    -------
    VGG11
        Loaded PyTorch Lightning model in evaluation mode.
    """
    model = VGG11.load_from_checkpoint(checkpoint_path, lr=lr)
    model.eval()
    return model


def predict(model, dataloader, device="cpu"):
    """
    Run inference on a dataset and return predicted labels.

    Parameters
    ----------
    model : VGG11
        PyTorch Lightning model for inference.
    dataloader : torch.utils.data.DataLoader
        DataLoader providing batches of input images.
    device : str, optional
        Device to run inference on ("cpu" or "cuda"; default is "cpu").

    Returns
    -------
    list of int
        Predicted class labels for all samples in the dataloader.
    """
    model.to(device)
    preds = []
    with torch.no_grad():
        for xb, _ in dataloader:
            xb = xb.to(device)
            out = model(xb)
            pred = out.argmax(1)
            preds.extend(pred.cpu().numpy())
    return preds


if __name__ == "__main__":
    checkpoint = "checkpoints/vgg11-smoker-epoch=02-val_acc=0.88.ckpt"
    dataloader = SmokerDataModule()
    model = load_model(checkpoint)
    #print("Model loaded for inference!")
