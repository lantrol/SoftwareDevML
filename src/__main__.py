import gradio as gr
from pathlib import Path
from PIL import Image
import random
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import plotly.express as px   # ✅ use Plotly instead of seaborn/matplotlib
from src.data_loader import SmokerDataModule  # replace with your actual import

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent  # one level up from /src
DATASET_DIR = BASE_DIR.parent / "data"
DATA_DIR = BASE_DIR / "data" #MODIFY

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

    # Create a subplot-like layout by combining plots using Gradio layout, not in a single figure
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

    return fig1, fig2, fig3a, fig3b  # return 4th as a list for flexible layout

# Gradio interface
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

        

if __name__ == "__main__":
    demo.launch()

