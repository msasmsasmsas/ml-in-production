#!/usr/bin/env python3

"""
Image classifier model implementation.
"""

import torch
from torchvision import models
from typing import Dict, List, Any, Union
import numpy as np

class ImageClassifier:
    """ResNet50 image classifier model."""

    def __init__(self, model_name: str = "resnet50", use_gpu: bool = False):
        """
        Initialize the classifier.

        Args:
            model_name: Name of the model to use
            use_gpu: Whether to use GPU for inference
        """
        self.name = model_name
        self.version = "1.0.0"
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")

        # Load the model
        if model_name == "resnet50":
            self.model = models.resnet50(pretrained=True)
        elif model_name == "mobilenet_v2":
            self.model = models.mobilenet_v2(pretrained=True)
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        # Move model to device and set to evaluation mode
        self.model = self.model.to(self.device)
        self.model.eval()

    def predict(self, image_data: np.ndarray) -> np.ndarray:
        """
        Run inference on the image data.

        Args:
            image_data: Preprocessed image data as numpy array
                        with shape (batch_size, channels, height, width)

        Returns:
            Model predictions as numpy array
        """
        # Convert numpy array to torch tensor with explicit dtype
        input_tensor = torch.from_numpy(image_data).float().to(self.device)

        # Run inference
        with torch.no_grad():
            output = self.model(input_tensor)
            # Apply softmax to get probabilities
            probabilities = torch.nn.functional.softmax(output, dim=1)

        # Convert to numpy array and return
        return probabilities.cpu().numpy()

    def get_metadata(self) -> Dict[str, Any]:
        """
        Get model metadata.
        Returns:
            Dictionary with model metadata
        """
        return {
            "name": self.name,
            "version": self.version,
            "framework": "PyTorch",
            "device": str(self.device),
            "input_shape": [1, 3, 224, 224],
            "output_shape": [1, 1000]
        }
