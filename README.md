# SoftwareDevML

## Authors

Lander Jiménez & María Goicoechea

## Dataset used

[Smoker Detection [Image] classification Dataset (Version 5)](https://www.kaggle.com/datasets/sujaykapadnis/smoking)

## Requirements

 ```
Python = 3.11
torch: 2.8.0+cpu        
torchvision: 0.23.0+cpu 
pytorch_lightning: 2.5.5
matplotlib: 3.10.6      
requests: 2.32.5   
tqdm: 4.67.1 
```

## How to get it working

1. Run src/dataset_download.py, that will download and organise the dataset into a data folder.
    - Optionally you can run dataset_loader_tester.py if you want to take a peek at what the dataset contains
2. Run src/model.py to start classifying.
    - It will output the results on the console and create a logs folder-
