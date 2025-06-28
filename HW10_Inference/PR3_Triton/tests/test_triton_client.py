#!/usr/bin/env python3

"""
Tests for the Triton Inference Server client.
"""

import unittest
import numpy as np
from unittest.mock import patch, MagicMock
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from client.client import preprocess_image, infer

class TestTritonClient(unittest.TestCase):

    @patch('client.client.Image.open')
    def test_preprocess_image(self, mock_image_open):
        # Mock image setup
        mock_img = MagicMock()
        mock_img.convert.return_value = mock_img
        mock_img.resize.return_value = mock_img
        mock_image_open.return_value = mock_img

        # Mock numpy conversion
        mock_array = np.zeros((224, 224, 3), dtype=np.uint8)
        np.array = MagicMock(return_value=mock_array)

        # Test function
        result = preprocess_image('test.jpg')

        # Assertions
        mock_image_open.assert_called_once_with('test.jpg')
        mock_img.convert.assert_called_once_with('RGB')
        mock_img.resize.assert_called_once_with((224, 224))

        # Check shape after preprocessing
        self.assertEqual(result.shape, (1, 3, 224, 224))

    @patch('client.client.httpclient.InferenceServerClient')
    def test_infer(self, mock_client_class):
        # Mock client
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.is_model_ready.return_value = True

        # Mock preprocess_image
        with patch('client.client.preprocess_image') as mock_preprocess:
            mock_preprocess.return_value = np.zeros((1, 3, 224, 224), dtype=np.float32)

            # Mock results
            mock_results = MagicMock()
            mock_output = np.zeros((1, 1000), dtype=np.float32)
            # Set some values to get predictable top indices
            mock_output[0, 5] = 0.8  # Highest score
            mock_output[0, 2] = 0.6  # Second highest
            mock_output[0, 9] = 0.4  # Third
            mock_results.as_numpy.return_value = mock_output
            mock_client.infer.return_value = mock_results

            # Mock ImageNet classes
            with patch('client.client.IMAGENET_CLASSES', {"5": "electric ray", "2": "great white shark", "9": "ostrich"}):
                # Call function
                result = infer('test.jpg', 'resnet50')

            # Assertions
            mock_client.is_model_ready.assert_called_once_with('resnet50')
            mock_client.infer.assert_called_once()

            # Check top prediction
            self.assertEqual(result[0]['class'], "electric ray")
            self.assertAlmostEqual(result[0]['probability'], 0.8)

if __name__ == '__main__':
    unittest.main()
