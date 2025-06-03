import unittest
from unittest.mock import patch, MagicMock
import io
import json
import streamlit as st
from PIL import Image
import numpy as np
import sys
import os

# Add the parent directory to path to import the app module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock streamlit
class StreamlitMock:
    def __init__(self):
        self.sidebar_items = []
        self.items = []
        self.button_clicked = False
        self.uploaded_file = None
        self.confidence = 0.5

    def set_page_config(self, **kwargs):
        pass

    def title(self, text):
        self.items.append(("title", text))

    def header(self, text):
        self.items.append(("header", text))

    def subheader(self, text):
        self.items.append(("subheader", text))

    def markdown(self, text):
        self.items.append(("markdown", text))

    def write(self, text):
        self.items.append(("write", text))

    def info(self, text):
        self.items.append(("info", text))

    def success(self, text):
        self.items.append(("success", text))

    def error(self, text):
        self.items.append(("error", text))

    def sidebar(self):
        return self

    def file_uploader(self, label, type):
        return self.uploaded_file

    def slider(self, label, min_value, max_value, value, step):
        return self.confidence

    def button(self, label):
        return self.button_clicked

    def spinner(self, text):
        class SpinnerContext:
            def __enter__(self_context):
                pass
            def __exit__(self_context, exc_type, exc_val, exc_tb):
                pass
        return SpinnerContext()

    def image(self, img, caption=None, use_column_width=None):
        self.items.append(("image", img, caption))

    def columns(self, n):
        class Column:
            def __init__(self):
                self.items = []
            def markdown(self, text):
                self.items.append(("markdown", text))
            def table(self, df):
                self.items.append(("table", df))
            def success(self, text):
                self.items.append(("success", text))
            def info(self, text):
                self.items.append(("info", text))
        return [Column() for _ in range(n)]
    
    def expander(self, label):
        class Expander:
            def __init__(self):
                self.items = []
            def json(self, data):
                self.items.append(("json", data))
        return Expander()

    def json(self, data):
        self.items.append(("json", data))

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class TestStreamlitApp(unittest.TestCase):
    
    @patch('streamlit.file_uploader')
    @patch('streamlit.button')
    @patch('requests.post')
    def test_analysis_with_no_threats(self, mock_post, mock_button, mock_file_uploader):
        # Create a test image
        image = Image.new('RGB', (100, 100), color='white')
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='JPEG')
        
        # Mock the file uploader to return the image
        uploaded_file = MagicMock()
        uploaded_file.read.return_value = img_byte_arr.getvalue()
        mock_file_uploader.return_value = uploaded_file
        
        # Mock button click
        mock_button.return_value = True
        
        # Mock API response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "threats": [],
            "recommendations": ["Keep monitoring your crops regularly."]
        }
        mock_post.return_value = mock_response
        
        # Run the test with mocked streamlit
        with patch('streamlit.image'), \
             patch('streamlit.spinner') as mock_spinner, \
             patch('streamlit.success') as mock_success, \
             patch('streamlit.info') as mock_info:
            
            # Mock context manager for spinner
            mock_spinner_context = MagicMock()
            mock_spinner.return_value.__enter__.return_value = mock_spinner_context
            
            # Import app here to use the mocked streamlit
            import app
            
            # Verify that the success message for no threats is displayed
            mock_success.assert_called_with("No threats detected!")
            
            # Verify that recommendations are displayed
            mock_info.assert_called_with("Keep monitoring your crops regularly.")

    @patch('streamlit.file_uploader')
    @patch('streamlit.button')
    @patch('requests.post')
    def test_analysis_with_threats(self, mock_post, mock_button, mock_file_uploader):
        # Create a test image
        image = Image.new('RGB', (100, 100), color='white')
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='JPEG')
        
        # Mock the file uploader to return the image
        uploaded_file = MagicMock()
        uploaded_file.read.return_value = img_byte_arr.getvalue()
        mock_file_uploader.return_value = uploaded_file
        
        # Mock button click
        mock_button.return_value = True
        
        # Mock API response with threats
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
        
        # Run the test with mocked streamlit
        with patch('streamlit.image'), \
             patch('streamlit.spinner') as mock_spinner, \
             patch('streamlit.table') as mock_table, \
             patch('streamlit.info') as mock_info, \
             patch('streamlit.expander') as mock_expander:
            
            # Mock context manager for spinner
            mock_spinner_context = MagicMock()
            mock_spinner.return_value.__enter__.return_value = mock_spinner_context
            
            # Import app here to use the mocked streamlit
            import app
            
            # Verify API was called with correct parameters
            mock_post.assert_called()
            
            # Note: Complete verification would require more complex mocking
            # of the streamlit components like columns and expander

    @patch('streamlit.file_uploader')
    @patch('streamlit.button')
    @patch('requests.post')
    def test_api_error(self, mock_post, mock_button, mock_file_uploader):
        # Create a test image
        image = Image.new('RGB', (100, 100), color='white')
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='JPEG')
        
        # Mock the file uploader to return the image
        uploaded_file = MagicMock()
        uploaded_file.read.return_value = img_byte_arr.getvalue()
        mock_file_uploader.return_value = uploaded_file
        
        # Mock button click
        mock_button.return_value = True
        
        # Mock API error
        mock_post.side_effect = Exception("API connection error")
        
        # Run the test with mocked streamlit
        with patch('streamlit.image'), \
             patch('streamlit.spinner') as mock_spinner, \
             patch('streamlit.error') as mock_error:
            
            # Mock context manager for spinner
            mock_spinner_context = MagicMock()
            mock_spinner.return_value.__enter__.return_value = mock_spinner_context
            
            # Import app here to use the mocked streamlit
            import app
            
            # Verify that error message is displayed
            mock_error.assert_called_with("Error connecting to the API: API connection error")


if __name__ == "__main__":
    unittest.main()
