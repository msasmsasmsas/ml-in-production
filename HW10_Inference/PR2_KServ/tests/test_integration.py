#!/usr/bin/env python3

"""
Integration tests for KServe deployment.
These tests may be skipped if the service is not available.
"""

import unittest
import os
import sys
import json
from PIL import Image
import time
import requests
from unittest.mock import patch

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from client.client import KServeClient

class TestKServeIntegration(unittest.TestCase):
    """Integration tests for KServe deployment."""

    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        # Configuration from environment variables
        cls.host = os.environ.get("KSERVE_HOST", "localhost")
        cls.port = int(os.environ.get("KSERVE_PORT", "8080"))
        cls.service = os.environ.get("KSERVE_SERVICE", "resnet50-classifier")
        cls.namespace = os.environ.get("KSERVE_NAMESPACE", "kserve-demo")

        # Create client
        cls.client = KServeClient(
            service_name=cls.service,
            namespace=cls.namespace,
            host=cls.host,
            port=cls.port
        )

        # Create a test image
        cls.test_image_path = os.path.join(os.path.dirname(__file__), "test_image.jpg")
        test_image = Image.new("RGB", (224, 224), color="blue")
        test_image.save(cls.test_image_path)

        # Check if the service is available
        cls.service_available = cls._check_service_available(cls)
        if not cls.service_available:
            print("Warning: KServe service is not available. Tests will be skipped.")

    @classmethod
    def tearDownClass(cls):
        """Clean up test environment."""
        # Remove test image
        if os.path.exists(cls.test_image_path):
            os.remove(cls.test_image_path)

    def _check_service_available(self):
        """Check if the KServe service is available."""
        try:
            # Try to connect to the health endpoint
            url = f"http://{self.host}:{self.port}/v1/models/{self.service}"
            response = requests.get(url, timeout=5)
            return response.status_code == 200
        except (requests.exceptions.RequestException, requests.exceptions.Timeout):
            return False

    def test_client_initialization(self):
        """Test client initialization."""
        self.assertEqual(self.client.service_name, self.service)
        self.assertEqual(self.client.namespace, self.namespace)
        self.assertEqual(self.client.host, self.host)
        self.assertEqual(self.client.port, self.port)

        expected_url = f"http://{self.host}:{self.port}/v1/models/{self.service}"
        self.assertEqual(self.client.base_url, expected_url)

    @unittest.skipIf(not getattr(TestKServeIntegration, "service_available", False),
                    "KServe service is not available")
    def test_health(self):
        """Test health check endpoint."""
        result = self.client.health()
        self.assertIn("status", result)
        self.assertEqual(result["status"], "healthy")

    @unittest.skipIf(not getattr(TestKServeIntegration, "service_available", False),
                    "KServe service is not available")
    def test_metadata(self):
        """Test metadata endpoint."""
        result = self.client.metadata()
        self.assertIn("name", result)
        self.assertEqual(result["name"], self.service)
        self.assertIn("versions", result)

    @unittest.skipIf(not getattr(TestKServeIntegration, "service_available", False),
                    "KServe service is not available")
    def test_predict(self):
        """Test predict endpoint."""
        # If service is available but we don't want to make actual calls during testing
        # we can mock the requests.post method
        with patch('requests.post') as mock_post:
            # Create mock response
            mock_response = unittest.mock.MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "predictions": [
                    [
                        {"rank": 1, "class_id": 0, "class_name": "tench", "probability": 0.9},
                        {"rank": 2, "class_id": 1, "class_name": "goldfish", "probability": 0.05}
                    ]
                ]
            }
            mock_post.return_value = mock_response

            # Call predict
            result = self.client.predict(self.test_image_path, topk=2)

            # Check results
            self.assertIn("predictions", result)
            self.assertIn("timing", result)

            # Check that request was made correctly
            mock_post.assert_called_once()
            args, kwargs = mock_post.call_args
            self.assertEqual(args[0], f"{self.client.base_url}/infer")
            self.assertIn("json", kwargs)

            # Check request payload
            payload = kwargs["json"]
            self.assertIn("instances", payload)
            self.assertEqual(len(payload["instances"]), 1)
            self.assertIn("data", payload["instances"][0])
            self.assertIn("b64", payload["instances"][0]["data"])
            self.assertIn("parameters", payload["instances"][0])
            self.assertEqual(payload["instances"][0]["parameters"]["topk"], 2)

    @unittest.skipIf(not getattr(TestKServeIntegration, "service_available", False),
                    "KServe service is not available")
    def test_explain(self):
        """Test explain endpoint."""
        # Mock the requests.post method
        with patch('requests.post') as mock_post:
            # Create mock response
            mock_response = unittest.mock.MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "explanations": [
                    {
                        "predictions": [
                            {"class_id": 0, "class_name": "tench", "probability": 0.9}
                        ],
                        "explanation_type": "feature_importance",
                        "explanation_data": {"message": "Feature importance data"}
                    }
                ]
            }
            mock_post.return_value = mock_response

            # Call explain
            result = self.client.explain(self.test_image_path)

            # Check results
            self.assertIn("explanations", result)
            self.assertIn("timing", result)

            # Check that request was made correctly
            mock_post.assert_called_once()
            args, kwargs = mock_post.call_args
            self.assertEqual(args[0], f"{self.client.base_url}/explain")
            self.assertIn("json", kwargs)

            # Check request payload
            payload = kwargs["json"]
            self.assertIn("instances", payload)
            self.assertEqual(len(payload["instances"]), 1)
            self.assertIn("data", payload["instances"][0])
            self.assertIn("b64", payload["instances"][0]["data"])
            self.assertIn("metadata", payload["instances"][0])
            self.assertTrue(payload["instances"][0]["metadata"]["explain"])

if __name__ == "__main__":
    unittest.main()
