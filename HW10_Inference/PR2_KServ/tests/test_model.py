#!/usr/bin/env python3

"""
Unit tests for the ResNet50 model implementation.
"""

import unittest
import torch
import sys
import os
import json
from unittest.mock import patch, MagicMock

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.model import ResNet50Model

class TestResNet50Model(unittest.TestCase):
    """Tests for the ResNet50Model class."""

    @classmethod
    def setUpClass(cls):
        """Set up the model once for all tests."""
        # Mock torch.device to avoid GPU checks
        with patch('torch.device', return_value='cpu'):
            # Mock models.resnet50 to avoid loading actual model
            with patch('torchvision.models.resnet50') as mock_model:
                # Create a mock model instance
                mock_instance = MagicMock()
                mock_model.return_value = mock_instance

                # Initialize the model
                cls.model = ResNet50Model()

                # Store the mock model for later assertions
                cls.mock_model_instance = mock_instance

    def test_initialization(self):
        """Test model initialization."""
        self.assertEqual(self.model.name, "resnet50")
        self.assertEqual(self.model.version, "1.0.0")
        self.assertTrue(hasattr(self.model, "model"))
        self.assertTrue(hasattr(self.model, "class_labels"))

        # Check that model was set to eval mode
        self.mock_model_instance.eval.assert_called_once()

    def test_load_imagenet_labels(self):
        """Test loading of ImageNet labels."""
        # Set up a temporary JSON file with test labels
        test_labels_file = os.path.join(os.path.dirname(__file__), "test_labels.json")
        test_labels = {
            "0": "test_class_0",
            "1": "test_class_1"
        }

        try:
            # Write test labels to file
            with open(test_labels_file, "w") as f:
                json.dump(test_labels, f)

            # Patch the path to use our test file
            with patch('os.path.dirname', return_value=os.path.dirname(__file__)):
                with patch('os.path.join', return_value=test_labels_file):
                    # Call the method
                    labels = self.model._load_imagenet_labels()

                    # Check results
                    self.assertEqual(labels, test_labels)
        finally:
            # Clean up test file
            if os.path.exists(test_labels_file):
                os.remove(test_labels_file)

    def test_predict(self):
        """Test the predict method."""
        # Create dummy input tensor
        dummy_input = torch.randn(2, 3, 224, 224)  # Batch of 2 images

        # Mock the model output
        dummy_output = torch.randn(2, 1000)  # Output for 1000 classes
        self.mock_model_instance.return_value = dummy_output

        # Mock softmax output
        with patch('torch.nn.functional.softmax', return_value=torch.tensor([[0.8, 0.1, 0.1], [0.7, 0.2, 0.1]])):
            # Mock topk to return known values
            with patch('torch.topk') as mock_topk:
                # Set up mock topk to return predictable values
                mock_topk.side_effect = lambda x, k: (torch.tensor([0.8, 0.1]), torch.tensor([0, 1]))

                # Call predict
                predictions, output = self.model.predict(dummy_input)

                # Check results
                self.assertEqual(len(predictions), 2)  # Two items in batch
                self.assertEqual(len(predictions[0]), 5)  # Top 5 predictions

                # Check structure of predictions
                pred = predictions[0][0]  # First prediction of first item
                self.assertIn("rank", pred)
                self.assertIn("class_id", pred)
                self.assertIn("class_name", pred)
                self.assertIn("probability", pred)

    def test_explain(self):
        """Test the explain method."""
        # Create dummy input tensor
        dummy_input = torch.randn(1, 3, 224, 224)  # Single image

        # Mock predict method to return known values
        with patch.object(self.model, 'predict') as mock_predict:
            mock_predict.return_value = ([[
                {"rank": 1, "class_id": 0, "class_name": "test", "probability": 0.9}
            ]], None)

            # Call explain
            explanation = self.model.explain(dummy_input)

            # Check results
            self.assertIsInstance(explanation, dict)
            self.assertIn("predictions", explanation)
            self.assertIn("explanation_type", explanation)
            self.assertIn("explanation_data", explanation)

    def test_get_metadata(self):
        """Test the get_metadata method."""
        metadata = self.model.get_metadata()

        # Check metadata contents
        self.assertEqual(metadata["name"], "resnet50")
        self.assertEqual(metadata["version"], "1.0.0")
        self.assertEqual(metadata["framework"], "PyTorch")
        self.assertEqual(metadata["input_shape"], [1, 3, 224, 224])
        self.assertEqual(metadata["output_shape"], [1, 1000])

if __name__ == "__main__":
    unittest.main()
