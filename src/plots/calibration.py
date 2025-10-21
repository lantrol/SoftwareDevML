import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import brier_score_loss
from tqdm import tqdm
from pathlib import Path
from PIL import Image

from src.modeling.model import VGG11
from src.data_loader import SmokerDataModule

def simple_calibration_plot(model, dataloader, device='cuda', n_bins=10, gradio=False):
    """
    Generate a calibration plot (reliability diagram) for a trained model.

    Parameters
    ----------
    model : torch.nn.Module
        Trained PyTorch model.
    dataloader : torch.utils.data.DataLoader
        DataLoader providing input images and labels.
    device : str, optional
        Device to run inference on ("cpu" or "cuda"), by default 'cuda'.
    n_bins : int, optional
        Number of bins for the calibration plot, by default 10.
    gradio : bool, optional
        If True, return figure for embedding (do not save/show). Default is False.

    Returns
    -------
    dict
        If gradio=False: returns {'brier_score', 'predicted_probabilities', 'true_labels'}
    tuple
        If gradio=True: returns (fig, brier_score)
    """
    print("Collecting predictions...")

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

    # --- Determine binary or multi-class ---
    if y_prob.shape[1] == 2:
        pos_probs = y_prob[:, 1]
        fig, ax = plt.subplots(figsize=(8, 6))
        CalibrationDisplay.from_predictions(
            y_true, pos_probs, n_bins=n_bins, ax=ax, name="Model"
        )
        ax.set_title("Calibration Plot (Reliability Diagram)", fontsize=14)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        brier_score = brier_score_loss(y_true, pos_probs)

        if gradio:
            return fig, brier_score
        else:
            # Save plot
            BASE_DIR = Path(__file__).parent
            PROJECT_ROOT = BASE_DIR.parent.parent
            REPORTS_DIR = PROJECT_ROOT.parent / "reports" / "figures"
            REPORTS_DIR.mkdir(parents=True, exist_ok=True)
            save_path = REPORTS_DIR / "calibration_plot.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.show()
            print(f"Calibration plot saved to: {save_path}")

    else:
        n_classes = y_prob.shape[1]
        fig, axes = plt.subplots(1, n_classes, figsize=(6 * n_classes, 6))
        if n_classes == 1:
            axes = [axes]

        brier_scores = []
        for i in range(n_classes):
            binary_true = (y_true == i).astype(int)
            CalibrationDisplay.from_predictions(
                binary_true, y_prob[:, i], n_bins=n_bins, ax=axes[i], name=f"Class {i}"
            )
            axes[i].set_title(f"Calibration for Class {i}")
            axes[i].grid(True, alpha=0.3)
            brier_scores.append(brier_score_loss(binary_true, y_prob[:, i]))

        plt.tight_layout()
        brier_score = np.mean(brier_scores)

        if gradio:
            return fig, brier_score
        else:
            BASE_DIR = Path(__file__).parent
            PROJECT_ROOT = BASE_DIR.parent.parent
            REPORTS_DIR = PROJECT_ROOT.parent / "reports" / "figures"
            REPORTS_DIR.mkdir(parents=True, exist_ok=True)
            save_path = REPORTS_DIR / "calibration_plot_multiclass.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.show()
            print(f"Multi-class calibration plot saved to: {save_path}")

    print(f"Brier Score: {brier_score:.4f} (lower is better)")

    if not gradio:
        return {
            'brier_score': brier_score,
            'predicted_probabilities': y_prob,
            'true_labels': y_true
        }

def get_high_loss_samples(y_true, y_prob, top_k=5):
    """
    Identify samples with the highest per-sample Brier loss.

    Parameters
    ----------
    y_true : np.ndarray, shape (N,)
        True labels.
    y_prob : np.ndarray, shape (N, num_classes)
        Predicted probabilities for each class.
    top_k : int, optional
        Number of top-loss samples to return, by default 5.

    Returns
    -------
    tuple
        - top_indices : np.ndarray
            Indices of the top-k highest loss samples.
        - top_losses : np.ndarray
            Brier loss values for the top-k samples.
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

def show_high_loss_samples(model, dataloader, device='cuda', top_k=5, gradio=False):
    """
    Display the top-k samples with the highest Brier loss.

    Parameters
    ----------
    model : VGG11
        Trained PyTorch Lightning model.
    dataloader : torch.utils.data.DataLoader
        DataLoader providing input images and labels.
    device : str, optional
        Device to run inference on ("cpu" or "cuda"), by default 'cuda'.
    top_k : int, optional
        Number of highest-loss samples to display, by default 5.
    gradio : bool, optional
        If True, return images suitable for Gradio Gallery instead of plotting, by default False.

    Returns
    -------
    If gradio=True:
        list of PIL.Image objects for top-k high-loss samples
    Otherwise:
        Displays a Matplotlib figure
    """
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

    if gradio:
        # Convert top-k images to PIL for Gradio
        pil_images = []
        for idx in top_indices:
            img = images[idx].permute(1, 2, 0).numpy()  # [H,W,C]
            img = np.clip(img * 255, 0, 255).astype(np.uint8)  # scale to 0-255
            pil_images.append(Image.fromarray(img))
        return pil_images
    else:
        # Plot using matplotlib
        fig, axes = plt.subplots(1, top_k, figsize=(4*top_k, 4))
        if top_k == 1:
            axes = [axes]  # Ensure axes is iterable
        for i, ax in enumerate(axes):
            img = images[top_indices[i]].permute(1, 2, 0).numpy()
            img = np.clip(img, 0, 1)
            ax.imshow(img)
            ax.set_title(f"True: {y_true[top_indices[i]]}\nLoss: {top_losses[i]:.4f}")
            ax.axis('off')
        plt.tight_layout()
        plt.show()
