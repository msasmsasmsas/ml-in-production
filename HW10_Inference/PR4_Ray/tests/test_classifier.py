#!/usr/bin/env python3

"""
Tests for the image classifier model.
"""

import unittest
import numpy as np
import sys
import os
import torch

# Add app directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.models.classifier import ImageClassifier

class TestImageClassifier(unittest.TestCase):

    def setUp(self):
        # Initialize classifier for testing
        self.classifier = ImageClassifier(model_name="resnet50", use_gpu=False)

    def test_initialization(self):
        # Test initialization of classifier
        self.assertEqual(self.classifier.name, "resnet50")
        self.assertEqual(self.classifier.version, "1.0.0")
        self.assertTrue(hasattr(self.classifier, "model"))

        # Check that model is in eval mode
        self.assertFalse(self.classifier.model.training)

    def test_predict(self):
        # Create dummy input data
        dummy_input = np.random.rand(1, 3, 224, 224).astype(np.float32)

        # Run prediction
        predictions = self.classifier.predict(dummy_input)

        # Check output shape (batch size, num classes)
        self.assertEqual(predictions.shape, (1, 1000))

        # Check that output sums to approximately 1 (softmax output)
        self.assertAlmostEqual(np.sum(predictions[0]), 1.0, places=5)

        # Check that all probabilities are between 0 and 1
        self.assertTrue(np.all(predictions >= 0))
        self.assertTrue(np.all(predictions <= 1))

    def test_get_metadata(self):
        # Test metadata function
        metadata = self.classifier.get_metadata()

        # Check metadata contents
        self.assertEqual(metadata["name"], "resnet50")
        self.assertEqual(metadata["version"], "1.0.0")
        self.assertEqual(metadata["framework"], "PyTorch")
        self.assertEqual(metadata["input_shape"], [1, 3, 224, 224])
        self.assertEqual(metadata["output_shape"], [1, 1000])

if __name__ == "__main__":
    unittest.main()
