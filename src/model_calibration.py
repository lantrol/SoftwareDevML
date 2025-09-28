import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import brier_score_loss
from tqdm import tqdm
from pathlib import Path

from model import VGG11
from data_loader import SmokerDataModule

def simple_calibration_plot(model, dataloader, device='cuda', n_bins=10):
    """
    Simplified version that directly uses sklearn CalibrationDisplay with arrays
    """
    print("Collecting predictions...")
    
    # Collect all predictions and labels
    all_probs = []
    all_labels = []
    
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader):
            inputs, labels = batch
            inputs = inputs.to(device)
            
            logits = model(inputs)
            probs = torch.softmax(logits, dim=1)
            
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    y_prob = np.concatenate(all_probs, axis=0)
    y_true = np.concatenate(all_labels, axis=0)
    
    # For binary classification
    if y_prob.shape[1] == 2:
        pos_probs = y_prob[:, 1]
        
        # Create calibration display directly
        fig, ax = plt.subplots(figsize=(8, 6))
        CalibrationDisplay.from_predictions(
            y_true, pos_probs, n_bins=n_bins, ax=ax, name="Model"
        )
        plt.title("Calibration Plot (Reliability Diagram)", fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        BASE_DIR = Path(__file__).parent
        REPORTS_DIR = BASE_DIR.parent / "reports" / "figures"
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        
        save_path = REPORTS_DIR / "calibration_plot.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Calibration plot saved to: {save_path}")
        
        plt.show()
        
        # Calculate metrics
        brier_score = brier_score_loss(y_true, pos_probs)
        
    else:
        print("Multi-class calibration - showing calibration for each class:")
        fig, axes = plt.subplots(1, y_prob.shape[1], figsize=(6*y_prob.shape[1], 6))
        if y_prob.shape[1] == 1:
            axes = [axes]
            
        brier_scores = []
        for i in range(y_prob.shape[1]):
            binary_true = (y_true == i).astype(int)
            CalibrationDisplay.from_predictions(
                binary_true, y_prob[:, i], n_bins=n_bins, 
                ax=axes[i], name=f"Class {i}"
            )
            axes[i].set_title(f"Calibration for Class {i}")
            axes[i].grid(True, alpha=0.3)
            
            brier_scores.append(brier_score_loss(binary_true, y_prob[:, i]))
        
        BASE_DIR = Path(__file__).parent
        REPORTS_DIR = BASE_DIR.parent / "reports" / "figures"
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        
        save_path = REPORTS_DIR / "calibration_plot_multiclass.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Multi-class calibration plot saved to: {save_path}")
        
        plt.tight_layout()
        plt.show()
        brier_score = np.mean(brier_scores)
    
    print(f"Brier Score: {brier_score:.4f} (lower is better)")
    
    return {
        'brier_score': brier_score,
        'predicted_probabilities': y_prob,
        'true_labels': y_true
    }


# Example usage
if __name__ == "__main__":
    from pathlib import Path
    # Load your model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load checkpoint
    BASE_DIR = Path(__file__).parent
    CHECKPOINT_DIR = BASE_DIR.parent / "checkpoints"
    ckpt_path = CHECKPOINT_DIR / "vgg11-smoker-epoch=02-val_acc=0.88.ckpt"
    
    model = VGG11.load_from_checkpoint(ckpt_path)
    model = model.to(device)
    model.eval()
    
    # Load data
    data_module = SmokerDataModule(data_dir="../data", batch_size=32, num_workers=0)
    data_module.setup()
    
    # Analyze calibration
    results = simple_calibration_plot(model, data_module.test_dataloader(), device)