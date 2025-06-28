#!/usr/bin/env python3

"""
Tests for the router deployment.
"""

import pytest
import numpy as np
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch, MagicMock
import sys
import os
import io
from PIL import Image

# Add app directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.deployments.router import RouterDeployment

# Create a test image
def create_test_image(size=(224, 224)):
    """Create a test image for testing."""
    img = Image.new("RGB", size, color="red")
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="JPEG")
    img_bytes.seek(0)
    return img_bytes

@pytest.fixture
def mock_classifier():
    """Create a mock classifier deployment."""
    mock = AsyncMock()
    # Setup the remote method to return test predictions
    mock.remote.return_value = {
        "predictions": [
            {"class_id": 1, "class_name": "goldfish", "probability": 0.9},
            {"class_id": 2, "class_name": "great white shark", "probability": 0.05}
        ],
        "model_name": "resnet50",
        "model_version": "1.0.0"
    }
    return mock

@pytest.fixture
def client(mock_classifier):
    """Create a test client with the mock classifier."""
    router = RouterDeployment(mock_classifier)
    return TestClient(router.app)

def test_health_endpoint(client):
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_metadata_endpoint(client):
    """Test the metadata endpoint."""
    response = client.get("/metadata")
    assert response.status_code == 200
    metadata = response.json()

    # Check basic metadata fields
    assert "service" in metadata
    assert "version" in metadata
    assert "endpoints" in metadata

    # Check endpoints list
    endpoints = metadata["endpoints"]
    assert len(endpoints) == 3  # health, metadata, predict

    # Verify all endpoint paths are present
    endpoint_paths = [ep["path"] for ep in endpoints]
    assert "/health" in endpoint_paths
    assert "/metadata" in endpoint_paths
    assert "/predict" in endpoint_paths

def test_predict_endpoint(client, mock_classifier):
    """Test the predict endpoint."""
    # Create test image
    test_img = create_test_image()

    # Send prediction request
    response = client.post(
        "/predict",
        files={"file": ("test.jpg", test_img, "image/jpeg")},
        params={"top_k": 2}
    )

    # Check response
    assert response.status_code == 200
    results = response.json()

    # Verify results structure
    assert "predictions" in results
    assert "model_name" in results
    assert "model_version" in results
    assert "inference_time" in results

    # Check predictions
    predictions = results["predictions"]
    assert len(predictions) == 2
    assert predictions[0]["class_name"] == "goldfish"
    assert predictions[0]["probability"] == 0.9

    # Verify classifier was called
    mock_classifier.remote.assert_called_once()

def test_predict_with_invalid_file(client):
    """Test predict endpoint with invalid file."""
    # Send request with non-image file
    response = client.post(
        "/predict",
        files={"file": ("test.txt", io.BytesIO(b"not an image"), "text/plain")},
    )

    # Check for error response
    assert response.status_code == 400
    assert "File must be an image" in response.json()["detail"]

def test_preprocess_image():
    """Test the image preprocessing function."""
    # Create a RouterDeployment instance with a mock classifier
    router = RouterDeployment(AsyncMock())

    # Create a test image
    img = Image.new("RGB", (100, 100), color="blue")

    # Preprocess the image
    processed = router._preprocess_image(img)

    # Check output shape (batch, channels, height, width)
    assert processed.shape == (1, 3, 224, 224)

    # Check data type
    assert processed.dtype == np.float32

    # Check normalization (values should be between -3 and 3 after normalization)
    assert np.all(processed >= -3)
    assert np.all(processed <= 3)
