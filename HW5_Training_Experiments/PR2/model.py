# E:\ml-in-production\HW5_Training_Experiments\PR2\model.py
import torch
import torch.nn as nn
from torchvision import models


def create_model(num_classes, model_name="resnet18", pretrained=True):
    """
    Creates a model for classification
    """
    if model_name == "resnet18":
        # Create ResNet model
        model = models.resnet18(weights='IMAGENET1K_V1' if pretrained else None)
        # Replace the final layer
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return model