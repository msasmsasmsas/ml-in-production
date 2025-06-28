#!/usr/bin/env python3

"""
Integration tests for Ray Serve deployment.
These tests require a running Ray Serve deployment to pass.
"""

import unittest
import requests
import io
from PIL import Image
import numpy as np
import sys
import os
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from client.client import RayServeClient

class TestRayServeIntegration(unittest.TestCase):
    """Integration tests for the Ray Serve deployment."""

    @classmethod
    def setUpClass(cls):
        """Set up the test client and check server availability."""
        cls.base_url = "http://localhost:8000"
        cls.client = RayServeClient(cls.base_url)

        # Try to connect to the server
        max_retries = 5
        for i in range(max_retries):
            try:
                cls.client.health_check()
                break
            except requests.exceptions.ConnectionError:
                if i == max_retries - 1:
                    raise unittest.SkipTest("Ray Serve is not running. Skipping integration tests.")
                time.sleep(1)

    def test_health_check(self):
        """Test health check endpoint."""
        response = self.client.health_check()
        self.assertEqual(response["status"], "healthy")

    def test_metadata(self):
        """Test metadata endpoint."""
        metadata = self.client.get_metadata()
        self.assertIn("service", metadata)
        self.assertIn("version", metadata)
        self.assertIn("endpoints", metadata)

    def test_predict_with_test_image(self):
        """Test prediction with a generated test image."""
        # Create a test image
        img = Image.new("RGB", (224, 224), color="blue")
        img_path = "test_image.jpg"
        img.save(img_path)

        try:
            # Send prediction request
            result = self.client.predict(img_path)

            # Check response structure
            self.assertIn("predictions", result)
            self.assertIn("model_name", result)
            self.assertIn("model_version", result)
            self.assertIn("inference_time", result)

            # Check predictions
            predictions = result["predictions"]
            self.assertEqual(len(predictions), 5)  # Default top_k is 5

            # Each prediction should have these fields
            for pred in predictions:
                self.assertIn("class_id", pred)
                self.assertIn("class_name", pred)
                self.assertIn("probability", pred)

                # Probability should be between 0 and 1
                self.assertGreaterEqual(pred["probability"], 0)
                self.assertLessEqual(pred["probability"], 1)
        finally:
            # Clean up the test image
            if os.path.exists(img_path):
                os.remove(img_path)

if __name__ == "__main__":
    unittest.main()
