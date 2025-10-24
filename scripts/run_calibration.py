from smoking_prediction.modeling.model import VGG11
from smoking_prediction.data_loader import SmokerDataModule
from smoking_prediction.plots.calibration import simple_calibration_plot, show_high_loss_samples
import torch
from pathlib import Path

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint
    BASE_DIR = Path(__file__).parent
    CHECKPOINT_DIR = BASE_DIR.parent / "checkpoints"
    ckpt_path = CHECKPOINT_DIR / "vgg11-smoker-epoch=02-val_acc=0.88.ckpt"

    model = VGG11.load_from_checkpoint(ckpt_path).to(device).eval()

    # Load data
    data_module = SmokerDataModule(data_dir="../data", batch_size=32, num_workers=0)
    data_module.setup()

    # Run calibration analysis
    results = simple_calibration_plot(model, data_module.test_dataloader(), device)
    show_high_loss_samples(model, data_module.test_dataloader(), device, top_k=5)

if __name__ == "__main__":
    main()
