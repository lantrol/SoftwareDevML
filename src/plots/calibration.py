import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import brier_score_loss
from tqdm import tqdm
from pathlib import Path

from src.modeling.model import VGG11
from src.data_loader import SmokerDataModule

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
        PROJECT_ROOT = BASE_DIR.parent.parent        # src
        REPORTS_DIR = PROJECT_ROOT.parent / "reports" / "figures"
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
        PROJECT_ROOT = BASE_DIR.parent.parent        # src
        REPORTS_DIR = PROJECT_ROOT.parent / "reports" / "figures"
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

def get_high_loss_samples(y_true, y_prob, top_k=5):
    """
    Returns indices and details of the top_k samples with highest Brier loss.
    
    y_true: np.array of shape (N,) with true labels
    y_prob: np.array of shape (N, num_classes) with predicted probabilities
    """
    # For binary classification
    if y_prob.shape[1] == 2:
        # Compute Brier loss per sample: (p - y)^2
        pos_probs = y_prob[:, 1]
        per_sample_loss = (pos_probs - y_true) ** 2
        
    else:
        # Multi-class Brier loss per sample: sum((p - y_onehot)^2)
        N, C = y_prob.shape
        y_onehot = np.zeros_like(y_prob)
        y_onehot[np.arange(N), y_true] = 1
        per_sample_loss = np.sum((y_prob - y_onehot) ** 2, axis=1)
    
    # Get indices of top_k losses
    top_indices = np.argsort(-per_sample_loss)[:top_k]
    
    return top_indices, per_sample_loss[top_indices]

def show_high_loss_samples(model, dataloader, device='cuda', top_k=5):
    model.eval()
    all_probs = []
    all_labels = []
    all_images = []

    with torch.no_grad():
        for batch in dataloader:
            imgs, labels = batch
            imgs = imgs.to(device)
            logits = model(imgs)
            probs = torch.softmax(logits, dim=1)
            
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_images.append(imgs.cpu())
    
    y_prob = np.concatenate(all_probs, axis=0)
    y_true = np.concatenate(all_labels, axis=0)
    images = torch.cat(all_images, dim=0)

    top_indices, top_losses = get_high_loss_samples(y_true, y_prob, top_k=top_k)

    print("Top loss samples:")
    for idx, loss in zip(top_indices, top_losses):
        print(f"Index {idx}: Loss={loss:.4f}, True={y_true[idx]}, Pred={y_prob[idx]}")

    # Plot the top-k images
    fig, axes = plt.subplots(1, top_k, figsize=(4*top_k, 4))
    for i, ax in enumerate(axes):
        img = images[top_indices[i]].permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)  # if normalized
        ax.imshow(img)
        ax.set_title(f"True: {y_true[top_indices[i]]}\nLoss: {top_losses[i]:.4f}")
        ax.axis('off')
    plt.tight_layout()
    plt.show()
