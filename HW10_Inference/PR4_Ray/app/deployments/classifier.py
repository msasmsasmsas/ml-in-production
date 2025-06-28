#!/usr/bin/env python3

"""
Ray Serve deployment for image classification model.
"""

from ray import serve
import numpy as np
from typing import Dict, List, Any
import json
import os
import sys

# Додаємо корінь проекту до sys.path для правильного імпорту модулів
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), '.')))

from app.models.classifier import ImageClassifier

class ImageClassifierDeployment:
    def __init__(self):
        """Initialize the deployment."""
        # Load the classifier model
        self.model = ImageClassifier()

        # Load ImageNet class labels
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        try:
            # Спочатку спробуємо завантажити основний файл
            with open(os.path.join(script_dir, 'models', 'imagenet_classes.json'), 'r') as f:
                self.class_labels = json.load(f)
        except json.JSONDecodeError:
            print("ПОМИЛКА: Основний файл класів має неправильний формат JSON. Використовуємо альтернативний файл.")
            # Якщо основний файл пошкоджений, використовуємо альтернативний
            with open(os.path.join(script_dir, 'models', 'imagenet_dummy.json'), 'r') as f:
                self.class_labels = json.load(f)
        except FileNotFoundError:
            print("ПОМИЛКА: Файл з класами не знайдено. Створюємо базовий словник.")
            # Якщо файл взагалі відсутній, створюємо базовий словник
            self.class_labels = {str(i): f"class_{i}" for i in range(1000)}

    async def classify_image(self, image_data: np.ndarray, top_k: int = 5) -> Dict[str, Any]:
        """
        Classify an image and return top predictions.

        Args:
            image_data: Preprocessed image data as numpy array
            top_k: Number of top predictions to return

        Returns:
            Dictionary with predictions and metadata
        """
        # Run inference
        predictions = self.model.predict(image_data)

        # Get top k predictions
        indices = np.argsort(predictions[0])[-top_k:][::-1]
        probabilities = predictions[0][indices]

        # Format results
        results = []
        for i, idx in enumerate(indices):
            results.append({
                "class_id": int(idx),
                "class_name": self.class_labels.get(str(idx), f"Unknown class {idx}"),
                "probability": float(probabilities[i])
            })

        # Return formatted results
        return {
            "predictions": results,
            "model_name": self.model.name,
            "model_version": self.model.version
        }

    async def __call__(self, image_data: np.ndarray, top_k: int = 5) -> Dict[str, Any]:
        """Handle requests to the deployment."""
        return await self.classify_image(image_data, top_k)
