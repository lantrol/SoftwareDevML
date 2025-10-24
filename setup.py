from setuptools import setup, find_packages

setup(
    name="smoking_prediction",
    version="0.2.2",
    description="Gradio interface for a VG11 based smoking prediction model",
    packages=find_packages(),
    python_requires=">=3.11",
    install_requires=[
        "gradio",
        "pandas",
        "numpy",
        "torch",
        "torchvision",
        "pytorch_lightning",
        "tqdm",
        "requests",
        "plotly",
        "scikit-learn",
        "platformdirs",
        "matplotlib",
    ],
    entry_points={
        "console_scripts": [
            "smoking-prediction = smoking_prediction.run:run",
        ]
    },
    package_data={
        "smoking_prediction": ["checkpoints/*.ckpt", "data/*", "reports/data/*"]
    },
)
