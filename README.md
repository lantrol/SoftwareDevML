# SoftwareDevML

This is a project developped for the subject Software Development Oriented to Machine Learning taken in Machine Learning Master at Public University of Navarre (UPNa/ NUP).

The goal is to use a VGG11 to solve a classification problem to assess whether an image contains a smoker or not.

This repository will not receive any more support.

## Authors

Lander Jiménez & María Goicoechea

## Dataset used

It will be automatically downloaded when installing the package.
From Kaggle:
[Smoker Detection [Image] classification Dataset (Version 5)](https://www.kaggle.com/datasets/sujaykapadnis/smoking)

## Requirements

All are established on the pyproject.toml and will be automatically solved once you download the package.

## How to get it working

Execute:

``` bash
uvx --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ --from smoking_prediction smoking-prediction
```

On the terminal it will tell you what its doing and automatically launch a browser view in your local net where you can:
1. Interact with the dataset
2. Train your own model choosing your hyperparameters
3. View the performance of a trained model

## Report 

Found inside
