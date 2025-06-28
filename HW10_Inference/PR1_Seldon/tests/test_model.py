#!/usr/bin/env python3

"""
Unit tests for the ResNet50Classifier model.
"""

import unittest
import numpy as np
import sys
import os
from PIL import Image
import io

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.ResNet50Classifier import ResNet50Classifier

class TestResNet50Classifier(unittest.TestCase):
    """Tests for the ResNet50Classifier model."""

    @classmethod
    def setUpClass(cls):
        """Set up the model once for all tests."""
        cls.model = ResNet50Classifier()

        # Create a test image
        cls.test_image = Image.new("RGB", (224, 224), color="blue")
        img_byte_arr = io.BytesIO()
        cls.test_image.save(img_byte_arr, format="JPEG")
        cls.image_bytes = img_byte_arr.getvalue()

    def test_initialization(self):
        """Test model initialization."""
        self.assertEqual(self.model.name, "resnet50")
        self.assertEqual(self.model.version, "1.0.0")
        self.assertTrue(hasattr(self.model, "model"))
        self.assertTrue(hasattr(self.model, "preprocess"))
        self.assertTrue(hasattr(self.model, "class_labels"))

    def test_convert_to_image_pil(self):
        """Test conversion of PIL Image."""
        result = self.model._convert_to_image(self.test_image)
        self.assertIsInstance(result, Image.Image)
        self.assertEqual(result.size, (224, 224))

    def test_convert_to_image_bytes(self):
        """Test conversion of image bytes."""
        result = self.model._convert_to_image(self.image_bytes)
        self.assertIsInstance(result, Image.Image)

    def test_convert_to_image_numpy(self):
        """Test conversion of numpy array."""
        # Test (H, W, C) format
        img_array = np.array(self.test_image)
        result = self.model._convert_to_image(img_array)
        self.assertIsInstance(result, Image.Image)

        # Test (C, H, W) format
        img_array = np.array(self.test_image).transpose(2, 0, 1)
        result = self.model._convert_to_image(img_array)
        self.assertIsInstance(result, Image.Image)

        # Test (B, C, H, W) format
        img_array = np.array(self.test_image).transpose(2, 0, 1)[np.newaxis, ...]
        result = self.model._convert_to_image(img_array)
        self.assertIsInstance(result, Image.Image)

    def test_predict_pil_image(self):
        """Test prediction with PIL Image input."""
        result = self.model.predict(self.test_image)
        self._validate_prediction_result(result)

    def test_predict_numpy_array(self):
        """Test prediction with numpy array input."""
        img_array = np.array(self.test_image)
        result = self.model.predict(img_array)
        self._validate_prediction_result(result)

    def test_predict_bytes(self):
        """Test prediction with image bytes input."""
        result = self.model.predict(self.image_bytes)
        self._validate_prediction_result(result)

    def _validate_prediction_result(self, result):
        """Validate the structure of prediction results."""
        self.assertIsInstance(result, dict)
        self.assertIn("predictions", result)
        self.assertIn("model_name", result)
        self.assertIn("model_version", result)

        # Check predictions
        predictions = result["predictions"]
        self.assertIsInstance(predictions, list)
        self.assertEqual(len(predictions), 5)  # Top 5 predictions

        # Check the first prediction
        first_pred = predictions[0]
        self.assertIn("rank", first_pred)
        self.assertIn("class_id", first_pred)
        self.assertIn("class_name", first_pred)
        self.assertIn("probability", first_pred)

        # Check probability values
        for pred in predictions:
            self.assertGreaterEqual(pred["probability"], 0.0)
            self.assertLessEqual(pred["probability"], 1.0)

if __name__ == "__main__":
    unittest.main()
