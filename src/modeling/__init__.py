"""
Subpackage consisting of classes related to defining, training, loading and predicting with the model.

The defined CNN model is a modified version of the VG11, adapted for our classification purpose.
The pre-trained IMAGENET1K_V1 backbone is used.

Modules:
- **model**: Defines the main VG11 based CNN model.
- **predict**: Functions for loading and predicting a model checkpoint.
- **train**: Function for training the model. 
"""

__all__ = ["model", "predict", "train"]