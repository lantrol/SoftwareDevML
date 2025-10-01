import matplotlib.pyplot as plt
import pickle
import numpy as np
import os

def plot_training_metrics(metrics_file='training_metrics.pkl', save_path='training_metrics.png'):
    import matplotlib.pyplot as plt
import pickle
import numpy as np
import os

def plot_training_metrics(metrics_file='training_metrics.pkl', save_path='training_metrics.png'):
    """
    Plot training and validation losses and validation accuracy from a saved pickle file.

    Parameters
    ----------
    metrics_file : str, optional
        Path to the pickled metrics file (default 'training_metrics.pkl').
    save_path : str, optional
        Path to save the resulting plot image (default 'training_metrics.png').

    Side Effects
    ------------
    - Loads metrics from a pickle file.
    - Creates plots for training loss, validation loss, and validation accuracy.
    - Saves the figure to the specified path, creating directories if needed.
    - Displays the plot with matplotlib.
    - Prints final metrics summary and debug info.
    """
    # Load metrics data
    try:
        with open(metrics_file, 'rb') as f:
            metrics = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find {metrics_file}")
        print("Make sure you've run the training script first to generate the metrics file.")
        return
    
    train_losses = metrics['train_losses']
    val_losses = metrics['val_losses']
    val_accs = metrics['val_accs']
    
    # Handle potential mismatched lengths (validation might run at epoch 0)
    min_length = min(len(train_losses), len(val_losses), len(val_accs))
    
    # Trim all lists to the same length
    train_losses = train_losses[:min_length]
    val_losses = val_losses[:min_length]
    val_accs = val_accs[:min_length]

    print("Debug info:")
    print(f"Train losses length: {len(train_losses)}")
    print(f"Val losses length: {len(val_losses)}")
    print(f"Val accs length: {len(val_accs)}")
    print(f"Train losses: {train_losses}")
    print(f"Val losses: {val_losses}")
    print(f"Val accs: {val_accs}")
    
    epochs = range(1, min_length + 1)
    
    print(f"Plotting {min_length} epochs of data")
    print(f"Train losses: {len(train_losses)}, Val losses: {len(val_losses)}, Val accs: {len(val_accs)}")
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot losses
    ax1.plot(epochs, train_losses, 'bo-', label='Training Loss', linewidth=2, markersize=6)
    ax1.plot(epochs, val_losses, 'ro-', label='Validation Loss', linewidth=2, markersize=6)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epochs', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(epochs)
    
    # Plot accuracy
    ax2.plot(epochs, val_accs, 'go-', label='Validation Accuracy', linewidth=2, markersize=6)
    ax2.set_title('Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epochs', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(epochs)
    ax2.set_ylim([0, 1])  # Accuracy should be between 0 and 1
    
    plt.tight_layout()


    # Check for path
    # Get the directory
    directory = os.path.dirname(save_path)

    # Check if it exists
    if os.path.isdir(directory):
        print(f"Directory exists: {directory}")
    else:
        print(f"Directory does NOT exist: {directory}")
        # Create it if missing
        os.makedirs(directory, exist_ok=True)
        print(f"Directory created: {directory}")
    
    # Save the plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {save_path}")
    
    # Show the plot
    plt.show()
    
    # Print final metrics
    print(f"\nTraining Summary:")
    print(f"Final Training Loss: {train_losses[-1]:.4f}")
    print(f"Final Validation Loss: {val_losses[-1]:.4f}")
    print(f"Final Validation Accuracy: {val_accs[-1]:.4f}")
    print(f"Best Validation Accuracy: {max(val_accs):.4f} (Epoch {val_accs.index(max(val_accs)) + 1})")

def plot_custom_metrics(train_losses, val_losses, val_accs, save_path='custom_metrics.png'):
    """
    Plot training and validation metrics directly from lists.

    Parameters
    ----------
    train_losses : list of float
        Training losses per epoch.
    val_losses : list of float
        Validation losses per epoch.
    val_accs : list of float
        Validation accuracies per epoch.
    save_path : str, optional
        Path to save the plot (default 'custom_metrics.png').

    Side Effects
    ------------
    - Creates a plot of losses and accuracies.
    - Saves the plot to the specified path.
    - Displays the plot with matplotlib.
    """
    # Handle potential mismatched lengths
    min_length = min(len(train_losses), len(val_losses), len(val_accs))
    
    # Trim all lists to the same length
    train_losses = train_losses[:min_length]
    val_losses = val_losses[:min_length]
    val_accs = val_accs[:min_length]
    
    epochs = range(1, min_length + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot losses
    ax1.plot(epochs, train_losses, 'bo-', label='Training Loss', linewidth=2, markersize=6)
    ax1.plot(epochs, val_losses, 'ro-', label='Validation Loss', linewidth=2, markersize=6)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epochs', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(epochs)
    
    # Plot accuracy
    ax2.plot(epochs, val_accs, 'go-', label='Validation Accuracy', linewidth=2, markersize=6)
    ax2.set_title('Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epochs', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(epochs)
    ax2.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {save_path}")
    plt.show()

if __name__ == "__main__":
    # Paths to check
    metrics_path = os.path.join('reports', 'data', 'training_metrics.pkl')
    plot_path = os.path.join('reports', 'figures', 'training_metrics.png')

    # Check if metrics file exists
    if os.path.isfile(metrics_path):
        print(f"Metrics file exists: {metrics_path}")
    else:
        print(f"Metrics file does NOT exist: {metrics_path}")

    plot_training_metrics(metrics_file=metrics_path,
                           save_path=plot_path)