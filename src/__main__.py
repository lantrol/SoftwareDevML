import gradio as gr
from pathlib import Path
from PIL import Image
import random
import pandas as pd
import numpy as np
import os
import plotly.express as px   
import torch
import pickle
import matplotlib.pyplot as plt

from src.data_loader import SmokerDataModule  
from src.plots.calibration import simple_calibration_plot, show_high_loss_samples
from src.modeling.model import VGG11
from src.modeling.train import train_model
from src.modeling.predict import load_model, predict


# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent  # one level up from /src
DATASET_DIR = BASE_DIR / "data"
DATA_DIR = BASE_DIR / "data" #MODIFY
CHECKPOINTS_DIR = BASE_DIR / "checkpoints"

# Model
dm = SmokerDataModule(data_dir=str(DATASET_DIR), batch_size=32, num_workers=4) 
dm.setup()
current_model = None
current_ckpt = None
device = "cuda" if torch.cuda.is_available() else "cpu"

# function to update the global model when checkpoint changes
def update_global_model(ckpt_name):
    global current_model, current_ckpt
    if not ckpt_name:
        current_model = None
        current_ckpt = None
        return "No checkpoint selected"
    
    ckpt_path = CHECKPOINTS_DIR / ckpt_name
    current_model = load_model(ckpt_path)
    current_model.to(device)
    current_ckpt = ckpt_name
    return f"Loaded checkpoint: {ckpt_name}"

# Function to load sample images per class
def get_sample_images(split="train", n_samples=5):
    split_path = DATASET_DIR / split
    classes = ["smoking", "no_smoking"]
    images_dict = {}
    
    for cls in classes:
        cls_path = split_path / cls
        images = list(cls_path.glob("*"))
        sampled = random.sample(images, min(len(images), n_samples))
        pil_images = [Image.open(img).convert("RGB") for img in sampled]
        images_dict[cls] = pil_images
    
    return images_dict["smoking"], images_dict["no_smoking"]

# ---- Helper functions ----
def dataset_to_df(dataset, split_name):
    samples = []
    for path, _ in dataset.samples:
        fname = os.path.basename(path)
        samples.append({"name": fname, "split": split_name})
    return pd.DataFrame(samples)

def generate_plots():
    # Load metadata
    categories = pd.read_csv(DATA_DIR / "categories.csv")   # name, category
    genres = pd.read_csv(DATA_DIR / "genres.csv")           # filename, classification
    df_classes = pd.read_csv(DATA_DIR / "class.csv")
    df_genre = pd.read_csv(DATA_DIR / "genres.csv")

    # Fix column naming
    genres = genres.rename(columns={"filename": "name"})
    df_classification = pd.merge(categories, genres, on="name", how="inner")

    # Rename first, *then* select
    df_classification = df_classification.rename(columns={
        "classification": "genre",
        "category": "class"
    })
    df_classification = df_classification[["name", "genre", "class"]]

    # Load data module
    dm = SmokerDataModule(data_dir=str(DATASET_DIR), batch_size=32, num_workers=4)
    dm.setup()

    # Get split info
    df_train = dataset_to_df(dm.train_dataset, "train")
    df_val   = dataset_to_df(dm.val_dataset, "val")
    df_test  = dataset_to_df(dm.test_dataset, "test")
    df_splits = pd.concat([df_train, df_val, df_test], ignore_index=True)

    # Merge with metadata
    df = pd.merge(df_classification, df_splits, on="name", how="inner")

    # ---- Plot 1: Genre Distribution by Split ----
    fig1 = px.histogram(
        df, x="split", color="genre",
        title="Genre Distribution by Split",
        barmode="group"
    )

    # ---- Plot 2: Class Distribution by Split ----
    fig2 = px.histogram(
        df, x="split", color="class",
        title="Class Distribution by Split",
        barmode="group"
    )

    # ---- Plot 3: Genre Distribution by Category and Label ----
    df_classes["label"] = df_classes["label"].replace({0: "No Smoking", 1:"Smoking"})
    df_cls_genre = pd.merge(df_genre, df_classes, on="name", how="inner")

    # Create a subplot-like layout
    fig3a = px.histogram(
        df, x="class", color="genre",
        title="Genre Distribution By Category",
        barmode="group"
    )
    fig3b = px.histogram(
        df_cls_genre, x="label", color="genre",
        title="Genre Distribution By Label",
        barmode="group"
    )

    return fig1, fig2, fig3a, fig3b  

def compute_mean_images_from_dataset(dataset, class_names, n_samples=50, img_size=(250, 250)):
    imgs_by_class = {cls: [] for cls in class_names}

    # Randomly sample subset of dataset
    sample_indices = random.sample(range(len(dataset.samples)), min(len(dataset.samples), n_samples))
    for idx in sample_indices:
        path, class_idx = dataset.samples[idx]
        try:
            img = Image.open(path).convert("RGB").resize(img_size)
            img = np.array(img, dtype=np.float32) / 255.0
            imgs_by_class[class_names[class_idx]].append(img)
        except Exception as e:
            print(f"Skipping {path}: {e}")

    mean_images = {}
    for cls, imgs in imgs_by_class.items():
        if imgs:
            mean_img = np.mean(imgs, axis=0)
            mean_img = (mean_img * 255).astype(np.uint8)
            mean_images[cls] = Image.fromarray(mean_img)
    
    return mean_images

def generate_mean_train(n_train):
    class_names = ["smoking", "no_smoking"]
    dm = SmokerDataModule(data_dir=str(DATASET_DIR), batch_size=32, num_workers=4)
    dm.setup()
    mean_train = compute_mean_images_from_dataset(dm.train_dataset, class_names, n_samples=n_train)
    return mean_train["smoking"], mean_train["no_smoking"]

def generate_mean_val(n_val):
    class_names = ["smoking", "no_smoking"]
    dm = SmokerDataModule(data_dir=str(DATASET_DIR), batch_size=32, num_workers=4)
    dm.setup()
    mean_val = compute_mean_images_from_dataset(dm.val_dataset, class_names, n_samples=n_val)
    return mean_val["smoking"], mean_val["no_smoking"]

def generate_mean_test(n_test):
    class_names = ["smoking", "no_smoking"]
    dm = SmokerDataModule(data_dir=str(DATASET_DIR), batch_size=32, num_workers=4)
    dm.setup()
    mean_test = compute_mean_images_from_dataset(dm.test_dataset, class_names, n_samples=n_test)
    return mean_test["smoking"], mean_test["no_smoking"]
# ------------------------ MODEL TRAINING ---------------------------------------------
def run_training(batch_size, max_epochs, lr):
    ckpt_path, metrics_path = train_model(batch_size=batch_size, max_epochs=max_epochs, lr=lr)
    return f"✅ Training complete!\nCheckpoint: {ckpt_path}\nMetrics: {metrics_path}"


def list_checkpoints():
    ckpt_dir = "checkpoints"
    if not os.path.exists(ckpt_dir):
        return []
    return [f for f in os.listdir(ckpt_dir) if f.endswith(".ckpt")]


def load_metrics_global(ckpt_name):
    """
    Load training metrics for the specified checkpoint.

    Returns:
        info_text (str): Info message about checkpoint/metrics.
        acc_text (str): Markdown string showing final validation accuracy.
        loss_fig (plotly figure): Loss curves.
        acc_fig (plotly figure): Accuracy curve.
    """
    if not ckpt_name:
        return "No checkpoint selected", "", None, None
    
    metrics_path = os.path.join("reports", "data", f"{os.path.splitext(ckpt_name)[0]}_metrics.pkl")
    if not os.path.exists(metrics_path):
        return f"No metrics found for {ckpt_name}", "", None, None

    with open(metrics_path, "rb") as f:
        metrics = pickle.load(f)

    train_losses = metrics["train_losses"]
    val_losses = metrics["val_losses"]
    val_accs = metrics["val_accs"]
    final_val_acc = val_accs[-1] if val_accs else None

    min_len = min(len(train_losses), len(val_losses))
    epochs = list(range(1, min_len + 1))
    
    df_loss = pd.DataFrame({
        "Epoch": epochs + epochs,
        "Loss": train_losses[:min_len] + val_losses[:min_len],
        "Type": ["Train"] * min_len + ["Validation"] * min_len,
    })

    df_acc = pd.DataFrame({
        "Epoch": list(range(1, len(val_accs) + 1)),
        "Accuracy": val_accs,
        "Type": ["Validation"] * len(val_accs),
    })

    loss_fig = px.line(df_loss, x="Epoch", y="Loss", color="Type", markers=True, title="Loss Curves")
    acc_fig = px.line(df_acc, x="Epoch", y="Accuracy", color="Type", markers=True, title="Validation Accuracy")

    acc_text = (
        f"### 🏁 Final Validation Accuracy: **{final_val_acc:.2f}**"
        if final_val_acc is not None
        else "No accuracy data available."
    )

    return f"Metrics for {ckpt_name}", acc_text, loss_fig, acc_fig
# ----------------------MODEL CALIBRATION -----------------------------
def calibration_plot_global(n_bins):
    if current_model is None:
        return "No checkpoint loaded", None
    fig, brier = simple_calibration_plot(current_model, dm.val_dataloader(), device=device, n_bins=n_bins, gradio=True)
    return f"{brier:.4f}", fig

def high_loss_global(top_k):
    if current_model is None:
        return []
    return show_high_loss_samples(current_model, dm.val_dataloader(), device=device, top_k=top_k, gradio=True)
# ------------------ GRADIO INTERFACE ---------------------------------
with gr.Blocks() as demo:
    gr.Markdown("# 🖼️ Smoking Dataset Explorer")

    # -------- Global Checkpoint Selector (outside all tabs) --------
    with gr.Row():
        ckpt_dropdown = gr.Dropdown(
            choices=list_checkpoints(),
            label="Select Checkpoint File",
            interactive=True
        )
        refresh_btn = gr.Button("🔄 Refresh List")
        # Add this status textbox here (OPTIONAL but helpful)
    ckpt_status = gr.Textbox(label="Checkpoint Status", interactive=False)

    # refresh button updates the dropdown list
    refresh_btn.click(lambda: gr.update(choices=list_checkpoints()), outputs=ckpt_dropdown)
    # So it loads the model
    ckpt_dropdown.change(
        fn=update_global_model,
        inputs=[ckpt_dropdown],
        outputs=[ckpt_status]
    )
    
    with gr.Tab("Data Exploration"):
        split_selector = gr.Dropdown(
            choices=["train", "val", "test"],
            value="train",
            label="Select Dataset Split"
        )
        num_samples_slider = gr.Slider(
            minimum=3, maximum=10, value=5, step=1, label="Images per Class"
        )
        
        # Two main columns for smoking / no_smoking
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 🚬 Smoking")
                # Show 2 images per row for smaller thumbnails
                smoking_gallery = gr.Gallery(label="Smoking Samples", columns=3, height="auto")
            with gr.Column():
                gr.Markdown("### ❌ No Smoking")
                no_smoking_gallery = gr.Gallery(label="No Smoking Samples", columns=3, height="auto")
    
        # Connect inputs to function
        split_selector.change(fn=get_sample_images, inputs=[split_selector, num_samples_slider],
                            outputs=[smoking_gallery, no_smoking_gallery])
        num_samples_slider.change(fn=get_sample_images, inputs=[split_selector, num_samples_slider],
                                outputs=[smoking_gallery, no_smoking_gallery])
    
    # ------------------- Data Analysis Tab -------------------
    with gr.Tab("Data Analysis"):
        gr.Markdown("### 📊 Data Analysis")
        generate_btn = gr.Button("Generate Plots")

        with gr.Row():
            output1 = gr.Plot(label="Genre Distribution by Split")
            output2 = gr.Plot(label="Class Distribution by Split")

        with gr.Row():
            output3a = gr.Plot(label="Genre Distribution by Category")
            output3b = gr.Plot(label="Genre Distribution by Label")

        # Connect the button to the function
        generate_btn.click(
            fn=generate_plots,
            inputs=[],
            outputs=[output1, output2, output3a, output3b],
        )

    # ------------------- Mean Image Tab -------------------
    with gr.Tab("Mean Image"):
        gr.Markdown("### 🧮 Mean Images per Class & Split")
        gr.Markdown("Use the sliders to control how many images are averaged for each dataset split.")

        # Sliders
        with gr.Row():
            n_train_slider = gr.Slider(10, 200, value=50, step=10, label="Train Samples for Mean")
            n_val_slider = gr.Slider(10, 200, value=50, step=10, label="Val Samples for Mean")
            n_test_slider = gr.Slider(10, 200, value=50, step=10, label="Test Samples for Mean")

        # Image rows
        with gr.Row():
            with gr.Column():
                gr.Markdown("#### Train Mean Images")
                train_smoking_img = gr.Image(label="Smoking (Train)")
                train_no_smoking_img = gr.Image(label="No Smoking (Train)")
            with gr.Column():
                gr.Markdown("#### Val Mean Images")
                val_smoking_img = gr.Image(label="Smoking (Val)")
                val_no_smoking_img = gr.Image(label="No Smoking (Val)")
            with gr.Column():
                gr.Markdown("#### Test Mean Images")
                test_smoking_img = gr.Image(label="Smoking (Test)")
                test_no_smoking_img = gr.Image(label="No Smoking (Test)")

        n_train_slider.change(
            fn=generate_mean_train,
            inputs=[n_train_slider],
            outputs=[train_smoking_img, train_no_smoking_img]
        )

        n_val_slider.change(
            fn=generate_mean_val,
            inputs=[n_val_slider],
            outputs=[val_smoking_img, val_no_smoking_img]
        )

        n_test_slider.change(
            fn=generate_mean_test,
            inputs=[n_test_slider],
            outputs=[test_smoking_img, test_no_smoking_img]
        )

    # ----------------- Train model tab --------------------------
    with gr.Tab("🧠 Train New Model"):
        gr.Markdown("### Set Hyperparameters")

        batch_size = gr.Slider(8, 128, value=32, step=8, label="Batch Size")
        max_epochs = gr.Slider(1, 50, value=10, step=1, label="Max Epochs")
        lr = gr.Slider(1e-5, 1e-2, value=1e-3, step=1e-5, label="Learning Rate")

        train_btn = gr.Button("🚀 Start Training")
        train_output = gr.Textbox(label="Training Log")

        train_btn.click(run_training, inputs=[batch_size, max_epochs, lr], outputs=train_output)

    # ----------------- Model performance tab --------------------------
    with gr.Tab("📊 View Performance"):
        gr.Markdown("### Select a Trained Checkpoint")

        info_output = gr.Textbox(label="Model Info", interactive=False)
        acc_text = gr.Markdown("")

        with gr.Row():
            with gr.Column():
                loss_plot = gr.Plot(label="Loss Curves")
            with gr.Column():
                acc_plot = gr.Plot(label="Accuracy Curve")

        # Update plots whenever checkpoint changes
        ckpt_dropdown.change(
            fn=lambda ckpt_name: load_metrics_global(ckpt_name),
            inputs=[ckpt_dropdown],
            outputs=[info_output, acc_text, loss_plot, acc_plot]
        )


    # ------------------- Calibration Model Tab -------------------
    with gr.Tab("Calibration Model"):
        gr.Markdown("### 🔧 Model Calibration")
        
        # Row 1: Calibration plot slider
        with gr.Row():
            n_bins_slider = gr.Slider(5, 50, value=10, step=1, label="Calibration bins")
            top_k_slider = gr.Slider(1, 10, value=5, step=1, label="Top Loss Samples")

        with gr.Row():
            calibration_plot_output = gr.Plot(label="Calibration Plot")
            high_loss_gallery = gr.Gallery(label="Top High-Loss Samples", columns=3, height="auto")

        brier_output = gr.Textbox(label="Brier Score")

        # Connect sliders to functions
        n_bins_slider.change(fn=calibration_plot_global, inputs=[n_bins_slider], outputs=[brier_output, calibration_plot_output])
        top_k_slider.change(fn=high_loss_global, inputs=[top_k_slider], outputs=[high_loss_gallery])

        # Also update plots if checkpoint changes
        ckpt_dropdown.change(fn=calibration_plot_global, inputs=[n_bins_slider], outputs=[brier_output, calibration_plot_output])
        ckpt_dropdown.change(fn=high_loss_global, inputs=[top_k_slider], outputs=[high_loss_gallery])

    # ------------------- Predictions Tab -------------------
    with gr.Tab("🧪 Predictions"):
        gr.Markdown("### Make Predictions on a Dataset Split")

        split_selector = gr.Dropdown(
            choices=["train", "val", "test"],
            value="val",
            label="Dataset Split"
        )
        predict_btn = gr.Button("Predict")
        prediction_output = gr.Dataframe(headers=["Filename", "True Label", "Predicted Label"])

        def predict_global(split_name):
            if current_model is None:
                return pd.DataFrame([], columns=["Filename", "True Label", "Predicted Label"])

            if split_name == "train":
                dataloader = dm.train_dataloader()
            elif split_name == "val":
                dataloader = dm.val_dataloader()
            else:
                dataloader = dm.test_dataloader()

            preds = predict(current_model, dataloader, device=device)
            samples = dataloader.dataset.samples
            filenames = [os.path.basename(path) for path, _ in samples]
            true_labels = [dm.train_dataset.classes[idx] for _, idx in samples]

            return pd.DataFrame({
                "Filename": filenames,
                "True Label": true_labels,
                "Predicted Label": preds
            })

        predict_btn.click(fn=predict_global, inputs=[split_selector], outputs=[prediction_output])
    # ------------------- Auto Load on App Start -------------------
    demo.load(
        fn=get_sample_images,
        inputs=[split_selector, num_samples_slider],
        outputs=[smoking_gallery, no_smoking_gallery],
        queue=False
    )

    demo.load(
        fn=generate_plots,
        inputs=[],
        outputs=[output1, output2, output3a, output3b],
        queue=False
    )

    demo.load(
        fn=lambda: generate_mean_train(n_train_slider.value) + 
                generate_mean_val(n_val_slider.value) + 
                generate_mean_test(n_test_slider.value),
        inputs=[],
        outputs=[
            train_smoking_img, train_no_smoking_img,
            val_smoking_img, val_no_smoking_img,
            test_smoking_img, test_no_smoking_img
        ],
        queue=False
    )
        

if __name__ == "__main__":
    demo.launch()

