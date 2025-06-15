#!/usr/bin/env python3

"""
Unit tests for the ResNet50 handler implementation.
"""

import unittest
import torch
import io
import json
import sys
import os
from unittest.mock import patch, MagicMock
from PIL import Image
import base64

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.handler import ResNet50Handler

class TestResNet50Handler(unittest.TestCase):
    """Tests for the ResNet50Handler class."""

    def setUp(self):
        """Set up a new handler for each test."""
        # Create the handler
        self.handler = ResNet50Handler()

        # Create a test image
        self.test_image = Image.new("RGB", (224, 224), color="blue")
        img_byte_arr = io.BytesIO()
        self.test_image.save(img_byte_arr, format="JPEG")
        self.image_bytes = img_byte_arr.getvalue()
        self.base64_image = base64.b64encode(self.image_bytes).decode("utf-8")

    def test_initialization(self):
        """Test handler initialization."""
        # Create mock context
        mock_context = MagicMock()
        mock_context.system_properties = {
            "model_dir": "/tmp/models",
            "gpu_id": "0"
        }
        mock_context.manifest = {"model": {"serializedFile": "model.pt"}}

        # Patch ResNet50Model initialization
        with patch('model.handler.ResNet50Model') as mock_model_class:
            # Initialize the handler
            self.handler.initialize(mock_context)

            # Check that model was initialized
            mock_model_class.assert_called_once()

            # Check handler state
            self.assertTrue(self.handler.initialized)
            self.assertIsNotNone(self.handler.transform)

    def test_preprocess(self):
        """Test preprocessing of input data."""
        # Set up handler
        self.handler.initialized = True
        self.handler.device = "cpu"
        self.handler.transform = MagicMock(return_value=torch.zeros(3, 224, 224))
        self.handler._convert_to_image = MagicMock(return_value=self.test_image)

        # Create test request
        test_request = [{
            "data": self.image_bytes
        }]

        # Call preprocess
        result = self.handler.preprocess(test_request)

        # Check results
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape[0], 1)  # Batch size

        # Check that convert_to_image was called
        self.handler._convert_to_image.assert_called_once_with(self.image_bytes)

        # Check that transform was called
        self.handler.transform.assert_called_once()

    def test_inference(self):
        """Test inference method."""
        # Set up handler
        self.handler.initialized = True
        self.handler.model = MagicMock()
        self.handler.model.predict.return_value = ([
            [{"rank": 1, "class_id": 0, "class_name": "test", "probability": 0.9}]
        ], None)
        self.handler.model.explain.return_value = {"predictions": [{}], "explanation_type": "test"}

        # Test normal prediction
        self.handler.explain_mode = False
        batch = torch.zeros(1, 3, 224, 224)
        result = self.handler.inference(batch)

        # Check results
        self.assertEqual(result, [{"rank": 1, "class_id": 0, "class_name": "test", "probability": 0.9}])
        self.handler.model.predict.assert_called_once_with(batch)

        # Test explain mode
        self.handler.explain_mode = True
        self.handler.model.predict.reset_mock()
        result = self.handler.inference(batch)

        # Check results
        self.assertEqual(result, {"predictions": [{}], "explanation_type": "test"})
        self.handler.model.explain.assert_called_once_with(batch)

    def test_postprocess(self):
        """Test postprocessing of inference output."""
        # Set up handler
        self.handler.initialized = True
        self.handler.model = MagicMock()
        self.handler.model.name = "resnet50"
        self.handler.model.version = "1.0.0"

        # Test normal prediction output
        self.handler.explain_mode = False
        self.handler.topk = 2
        self.handler.start_time = 123456789.0

        # Create test predictions (batch of 2 items, 5 predictions each)
        test_predictions = [
            [{"rank": i+1, "class_id": i, "class_name": f"class_{i}", "probability": 0.9-i*0.1} for i in range(5)],
            [{"rank": i+1, "class_id": i+10, "class_name": f"class_{i+10}", "probability": 0.8-i*0.1} for i in range(5)]
        ]

        # Call postprocess
        result = self.handler.postprocess(test_predictions)

        # Check results
        self.assertEqual(len(result), 2)  # Two items in batch

        # Check first item
        self.assertEqual(len(result[0]["predictions"]), 2)  # Should be limited to topk=2
        self.assertEqual(result[0]["predictions"][0]["class_name"], "class_0")
        self.assertEqual(result[0]["model_name"], "resnet50")
        self.assertEqual(result[0]["model_version"], "1.0.0")

        # Test explain mode output
        self.handler.explain_mode = True
        test_explanation = {"predictions": [{"class_name": "test"}], "explanation_type": "feature_importance"}

        # Call postprocess
        result = self.handler.postprocess(test_explanation)

        # Check results
        self.assertEqual(len(result), 1)  # One response
        self.assertEqual(result[0], test_explanation)  # Should return explanation directly

    def test_convert_to_image_bytes(self):
        """Test conversion of bytes to image."""
        # Set up handler
        self.handler.initialized = True

        # Test with direct bytes
        with patch('PIL.Image.open', return_value=self.test_image) as mock_open:
            result = self.handler._convert_to_image(self.image_bytes)

            # Check results
            mock_open.assert_called_once()
            self.assertEqual(result, self.test_image)

    def test_convert_to_image_base64_dict(self):
        """Test conversion of base64 dict to image."""
        # Set up handler
        self.handler.initialized = True

        # Test with base64 in dict
        test_data = {"b64": self.base64_image}

        with patch('base64.b64decode', return_value=self.image_bytes) as mock_b64decode:
            with patch('PIL.Image.open', return_value=self.test_image) as mock_open:
                result = self.handler._convert_to_image(test_data)

                # Check results
                mock_b64decode.assert_called_once_with(self.base64_image)
                mock_open.assert_called_once()
                self.assertEqual(result, self.test_image)

    def test_is_explain_request(self):
        """Test detection of explanation requests."""
        # Test direct explain flag
        request = {"explain": True}
        self.assertTrue(self.handler._is_explain_request(request))

        # Test explain in metadata
        request = {"metadata": {"explain": True}}
        self.assertTrue(self.handler._is_explain_request(request))

        # Test non-explain request
        request = {"data": "some data"}
        self.assertFalse(self.handler._is_explain_request(request))

    def test_get_topk_param(self):
        """Test extraction of topk parameter."""
        # Test direct topk parameter
        request = {"topk": 10}
        self.assertEqual(self.handler._get_topk_param(request), 10)

        # Test topk in parameters
        request = {"parameters": {"topk": 15}}
        self.assertEqual(self.handler._get_topk_param(request), 15)

        # Test topk in metadata
        request = {"metadata": {"topk": 20}}
        self.assertEqual(self.handler._get_topk_param(request), 20)

        # Test default value
        request = {"data": "some data"}
        self.assertEqual(self.handler._get_topk_param(request), 5)

if __name__ == "__main__":
    unittest.main()
