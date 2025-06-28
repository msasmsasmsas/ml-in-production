#!/usr/bin/env python3

"""
ResNet50 model implementation for KServe/TorchServe.
"""

import torch
from torchvision import models
from typing import Dict, List, Any, Tuple, Optional
import json
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ResNet50Model:
    """ResNet50 image classifier model for KServe/TorchServe."""

    def __init__(self):
        """
        Initialize the ResNet50 model.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Load pre-trained ResNet50 model
        self.model = models.resnet50(pretrained=True)
        self.model.to(self.device)
        self.model.eval()

        # Load ImageNet class labels
        self.class_labels = self._load_imagenet_labels()

        # Model metadata
        self.name = "resnet50"
        self.version = "1.0.0"

        logger.info("ResNet50 model initialized successfully")

    def _load_imagenet_labels(self) -> Dict[int, str]:
        """
        Load ImageNet class labels from JSON file.

        Returns:
            Dictionary mapping class indices to class names
        """
        # Path to ImageNet class labels file
        labels_path = os.path.join(os.path.dirname(__file__), "imagenet_classes.json")

        try:
            if os.path.exists(labels_path):
                with open(labels_path, "r") as f:
                    return json.load(f)
            else:
                # Fallback to a small subset of labels if file not found
                logger.warning("ImageNet classes file not found. Using fallback labels.")
                return {
                    0: "tench",
                    1: "goldfish",
                    2: "great white shark",
                    # This is a truncated list for illustration
                }
        except Exception as e:
            logger.error(f"Error loading ImageNet labels: {e}")
            return {}

    def predict(self, input_tensor: torch.Tensor) -> Tuple[List[Dict[str, Any]], torch.Tensor]:
        """
        Run inference on the input tensor.

        Args:
            input_tensor: Input tensor with shape (batch_size, channels, height, width)

        Returns:
            Tuple containing:
            - List of prediction dictionaries (class name, probability, etc.)
            - Raw output tensor for explanations
        """
        # Move input to device
        input_tensor = input_tensor.to(self.device)

        # Run inference
        with torch.no_grad():
            output = self.model(input_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)

        # Process each item in the batch
        batch_predictions = []
        for i in range(probabilities.shape[0]):
            # Get top 5 predictions for this item
            probs, indices = torch.topk(probabilities[i], 5)
            probs = probs.cpu().numpy()
            indices = indices.cpu().numpy()

            # Format predictions
            predictions = []
            for j, (prob, idx) in enumerate(zip(probs, indices)):
                class_id = int(idx)
                class_name = self.class_labels.get(str(class_id), f"Unknown class {class_id}")
                predictions.append({
                    "rank": j + 1,
                    "class_id": class_id,
                    "class_name": class_name,
                    "probability": float(prob)
                })

            batch_predictions.append(predictions)

        return batch_predictions, output

    def explain(self, input_tensor: torch.Tensor) -> Dict[str, Any]:
        """
        Generate explanations for model predictions.

        Args:
            input_tensor: Input tensor with shape (batch_size, channels, height, width)

        Returns:
            Dictionary with explanation data
        """
        # This is a placeholder for model explanation
        # In a real implementation, you would integrate with LIME or other explanation methods

        # Get predictions
        predictions, _ = self.predict(input_tensor)

        # Return basic explanation
        return {
            "predictions": predictions,
            "explanation_type": "feature_importance",
            "explanation_data": {
                "message": "Feature importance visualization would be generated here"
            }
        }

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
            "output_shape": [1, 1000],
            "description": "ResNet50 image classifier pre-trained on ImageNet"
        }
