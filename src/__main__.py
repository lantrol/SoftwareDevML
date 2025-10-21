import gradio as gr
from pathlib import Path
from PIL import Image
import random
import pandas as pd
import numpy as np
import os
import plotly.express as px   
import torch

from src.data_loader import SmokerDataModule  
from src.plots.calibration import simple_calibration_plot_gradio, show_high_loss_samples_gradio
from src.modeling.model import VGG11


# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent  # one level up from /src
DATASET_DIR = BASE_DIR / "data"
DATA_DIR = BASE_DIR / "data" #MODIFY
CHECKPOINTS_DIR = BASE_DIR / "checkpoints"

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

# ------------------------ MODEL CALIBRATION ------------------------------------------
# --- Load model and data ---
device = "cuda" if torch.cuda.is_available() else "cpu"
dm = SmokerDataModule(data_dir=str(DATASET_DIR), batch_size=32, num_workers=4)
dm.setup()
model = VGG11.load_from_checkpoint(CHECKPOINTS_DIR / "vgg11-smoker-epoch=02-val_acc=0.88.ckpt")
model.to(device)

# --- Function to update calibration plot ---
def calibration_plot_gradio(n_bins):
    fig, brier = simple_calibration_plot_gradio(model, dm.val_dataloader(), device=device, n_bins=n_bins)
    return f"{brier:.4f}", fig

# --- Optional: function to get high-loss samples ---
def high_loss_samples_gradio(top_k):
    fig = show_high_loss_samples_gradio(model, dm.val_dataloader(), device=device, top_k=top_k)
    return fig


# ------------------ GRADIO INTERFACE ---------------------------------
with gr.Blocks() as demo:

    gr.Markdown("# 🖼️ Smoking Dataset Explorer")
    
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
    # ------------------- Calibration Model Tab -------------------
    with gr.Tab("Calibration Model"):
        gr.Markdown("### 🔧 Model Calibration")
        
        # Row 1: Calibration plot slider
        with gr.Row():
            n_bins_slider = gr.Slider(5, 50, value=10, step=1, label="Calibration bins")
            top_k_slider = gr.Slider(1, 10, value=5, step=1, label="Top Loss Samples")

        with gr.Row():
            calibration_plot_output = gr.Plot(label="Calibration Plot")
            high_loss_output = gr.Plot(label="Top High-Loss Samples")

        with gr.Row():
            brier_output = gr.Textbox(label="Brier Score")

        # --- Real-time plot updates ---
        n_bins_slider.change(
            fn=lambda n: calibration_plot_gradio(n),
            inputs=[n_bins_slider],
            outputs=[brier_output, calibration_plot_output],
        )

        top_k_slider.change(
            fn=show_high_loss_samples_gradio,
            inputs=[top_k_slider],
            outputs=[high_loss_output],
        )

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

