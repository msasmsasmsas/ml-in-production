import unittest
from unittest.mock import patch, MagicMock
import io
import json
import numpy as np
from PIL import Image
import pandas as pd
import sys
import os

# Add the parent directory to path to import the app module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the function to test
from app import predict_image

class TestGradioApp(unittest.TestCase):
    
    def setUp(self):
        # Create a test image
        self.test_image = Image.new('RGB', (100, 100), color='white')
    
    @patch('requests.post')
    def test_predict_image_success_with_threats(self, mock_post):
        # Mock successful API response with threats
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "threats": [
                {"type": "disease", "confidence": 0.85, "name": "Late Blight"},
                {"type": "pest", "confidence": 0.72, "name": "Aphids"}
            ],
            "recommendations": [
                "Apply fungicide for Late Blight control",
                "Use insecticide to manage Aphids"
            ],
            "details": {
                "severity": "moderate",
                "affected_area": "leaves"
            }
        }
        mock_post.return_value = mock_response
        
        # Call the function
        threats_table, recommendations, details = predict_image(self.test_image, 0.5)
        
        # Verify function called API correctly
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        self.assertEqual(kwargs['params'], {"confidence": 0.5})
        self.assertTrue('file' in kwargs['files'])
        
        # Verify outputs
        self.assertIn("disease", threats_table)
        self.assertIn("Late Blight", threats_table)
        self.assertIn("Apply fungicide", recommendations)
        self.assertIn("moderate", details)
    
    @patch('requests.post')
    def test_predict_image_success_without_threats(self, mock_post):
        # Mock successful API response with no threats
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "threats": [],
            "recommendations": ["Keep monitoring your crops regularly."],
            "details": {}
        }
        mock_post.return_value = mock_response
        
        # Call the function
        threats_table, recommendations, details = predict_image(self.test_image, 0.5)
        
        # Verify outputs
        self.assertEqual("No threats detected in this image.", threats_table)
        self.assertEqual("Keep monitoring your crops regularly.", recommendations)
        self.assertEqual("{}", details)
    
    @patch('requests.post')
    def test_predict_image_api_error(self, mock_post):
        # Mock API error
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_post.return_value = mock_response
        
        # Call the function
        threats_table, recommendations, details = predict_image(self.test_image, 0.5)
        
        # Verify error handling
        self.assertTrue(threats_table.startswith("Error:"))
        self.assertTrue(recommendations.startswith("API Error:"))
        self.assertEqual("", details)
    
    @patch('requests.post')
    def test_predict_image_exception(self, mock_post):
        # Mock exception during API call
        mock_post.side_effect = Exception("Connection refused")
        
        # Call the function
        threats_table, recommendations, details = predict_image(self.test_image, 0.5)
        
        # Verify error handling
        self.assertEqual("Error connecting to the API", threats_table)
        self.assertEqual("Details: Connection refused", recommendations)
        self.assertEqual("", details)
    
    @patch('requests.post')
    def test_confidence_threshold_passing(self, mock_post):
        # Mock successful API response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"threats": [], "recommendations": []}
        mock_post.return_value = mock_response
        
        # Call the function with different confidence values
        predict_image(self.test_image, 0.3)
        args1, kwargs1 = mock_post.call_args
        
        predict_image(self.test_image, 0.8)
        args2, kwargs2 = mock_post.call_args
        
        # Verify confidence was passed correctly
        self.assertEqual(kwargs1['params'], {"confidence": 0.3})
        self.assertEqual(kwargs2['params'], {"confidence": 0.8})


if __name__ == "__main__":
    unittest.main()
