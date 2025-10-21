import gradio as gr
from pathlib import Path
from PIL import Image
import random
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
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
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    sns.countplot(data=df, x="split", hue="genre", palette="colorblind", ax=ax1)
    ax1.set_title("Genre Distribution by Split")
    ax1.set_ylabel("Count")
    ax1.set_xlabel("Split")
    ax1.legend(title="Genre")
    plt.tight_layout()

    # ---- Plot 2: Class Distribution by Split ----
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.countplot(data=df, x="split", hue="class", palette="colorblind", ax=ax2)
    ax2.set_title("Class Distribution by Split")
    ax2.set_ylabel("Count")
    ax2.set_xlabel("Split")
    ax2.legend(title="Class")
    plt.tight_layout()

    # ---- Plot 3 & 4: Genre Distribution by Category and Label ----
    df_classes["label"] = df_classes["label"].replace({0: "No Smoking", 1:"Smoking"})
    df_cls_genre = pd.merge(df_genre, df_classes, on="name", how="inner")

    fig3, (ax3, ax4) = plt.subplots(1, 2, figsize=(12, 5))
    sns.countplot(data=df, x="class", hue="genre", palette="colorblind", ax=ax3)
    ax3.set_title("Genre Distribution By Category")
    ax3.set_ylabel("Count")
    ax3.set_xlabel("Class")
    ax3.legend(title="Genre")

    sns.countplot(data=df_cls_genre, x="label", hue="genre", palette="colorblind", ax=ax4)
    ax4.set_title("Genre Distribution By Label")
    ax4.set_ylabel("Count")
    ax4.set_xlabel("")
    ax4.legend(title="Genre")

    plt.tight_layout()
    
    return fig1, fig2, fig3

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
        """
        
        with gr.Row():
            with gr.Column():
                output1 = gr.Plot(label="Genre Distribution by Split")
            with gr.Column():
                output2 = gr.Plot(label="Class Distribution by Split")
            with gr.Column():
                output3 = gr.Plot(label="Genre Distribution by Category & Label")
        
        """


        generate_btn = gr.Button("Generate Plots")

        # Arrange plots in columns
        with gr.Row():
            with gr.Column():
                output1 = gr.Plot(label="Genre Distribution by Split")
            with gr.Column():
                output2 = gr.Plot(label="Class Distribution by Split")
            with gr.Column():
                output3 = gr.Plot(label="Genre Distribution by Category & Label")

        # Connect the button to the function
        generate_btn.click(
            fn=generate_plots,
            inputs=[],
            outputs=[output1, output2, output3]
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
        outputs=[output1, output2, output3],
        queue=False
    )

        

if __name__ == "__main__":
    demo.launch()

