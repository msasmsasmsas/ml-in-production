#!/usr/bin/env python3

"""
Integration tests for Seldon Core deployment.
"""

import unittest
import requests
import json
import os
import sys
import time
from PIL import Image

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from client.client import SeldonClient

class TestSeldonIntegration(unittest.TestCase):
    """Integration tests for the Seldon deployment."""

    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        # Configuration
        cls.host = os.environ.get("SELDON_HOST", "localhost")
        cls.port = int(os.environ.get("SELDON_PORT", "8003"))
        cls.deployment = os.environ.get("SELDON_DEPLOYMENT", "resnet50-classifier")
        cls.namespace = os.environ.get("SELDON_NAMESPACE", "seldon")
        cls.gateway = os.environ.get("SELDON_GATEWAY", "ambassador")

        # Create client
        cls.client = SeldonClient(
            deployment_name=cls.deployment,
            namespace=cls.namespace,
            host=cls.host,
            port=cls.port,
            gateway=cls.gateway
        )

        # Create a test image
        cls.test_image_path = os.path.join(os.path.dirname(__file__), "test_image.jpg")
        test_image = Image.new("RGB", (224, 224), color="blue")
        test_image.save(cls.test_image_path)

        # Check if the service is available
        cls.service_available = cls._check_service_available(cls)
        if not cls.service_available:
            print("Warning: Seldon service is not available. Some tests will be skipped.")

    @classmethod
    def tearDownClass(cls):
        """Clean up test environment."""
        # Remove test image
        if os.path.exists(cls.test_image_path):
            os.remove(cls.test_image_path)

    def _check_service_available(self):
        """Check if the Seldon service is available."""
        try:
            url = f"{self.client.base_url}/api/v1.0/health/status"
            response = requests.get(url, timeout=5)
            return response.status_code == 200
        except (requests.exceptions.RequestException, requests.exceptions.Timeout):
            return False

    def test_client_initialization(self):
        """Test client initialization."""
        self.assertEqual(self.client.deployment_name, self.deployment)
        self.assertEqual(self.client.namespace, self.namespace)
        self.assertEqual(self.client.host, self.host)
        self.assertEqual(self.client.port, self.port)
        self.assertEqual(self.client.gateway, self.gateway)

    @unittest.skipIf(not getattr(TestSeldonIntegration, "service_available", False),
                    "Seldon service is not available")
    def test_predict(self):
        """Test prediction endpoint."""
        result = self.client.predict(self.test_image_path)

        # Validate response structure
        self.assertIsInstance(result, dict)
        self.assertIn("data", result)

        # Extract predictions
        data = result["data"]
        self.assertIsInstance(data, dict)

        # Check for predictions or ndarray in data
        self.assertTrue(
            "predictions" in data or "ndarray" in data,
            "Response should contain 'predictions' or 'ndarray'"
        )

        # Validate inference time
        self.assertIn("inference_time", data)
        self.assertIsInstance(data["inference_time"], (int, float))

        # If predictions are present, validate them
        if "predictions" in data:
            predictions = data["predictions"]
            self.assertIsInstance(predictions, list)
            self.assertGreater(len(predictions), 0, "No predictions returned")

            # Validate first prediction
            first_pred = predictions[0]
            self.assertIsInstance(first_pred, dict)
            self.assertIn("class_name", first_pred)
            self.assertIn("probability", first_pred)

            # Check probability values
            for pred in predictions:
                self.assertGreaterEqual(pred["probability"], 0.0)
                self.assertLessEqual(pred["probability"], 1.0)

if __name__ == "__main__":
    unittest.main()
