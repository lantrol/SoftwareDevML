import torch
from src.modeling.model import VGG11
from src.data_loader import SmokerDataModule

def load_model(checkpoint_path, lr=1e-3):
    model = VGG11.load_from_checkpoint(checkpoint_path, lr=lr)
    model.eval()
    return model


def predict(model, dataloader, device="cpu"):
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
    checkpoint = "checkpoints/vgg11-smoker-epoch=02-val_acc=0.90.ckpt"
    dataloader = SmokerDataModule()
    model = load_model(checkpoint)
    #print("Model loaded for inference!")
