import pytest
from fastapi.testclient import TestClient
from app import app
from PIL import Image
import io
import numpy as np
import json

# Create test client
client = TestClient(app)

@pytest.fixture
def test_image():
    # Create a test image
    image = Image.new('RGB', (100, 100), color='white')
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    return img_byte_arr.getvalue()

def test_root_endpoint():
    """Test the root endpoint for health check"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "version" in data

def test_health_endpoint():
    """Test the health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "time" in data
    assert data["model_loaded"] is not None

def test_docs_info_endpoint():
    """Test the docs info endpoint"""
    response = client.get("/docs-info")
    assert response.status_code == 200
    data = response.json()
    assert "openapi_url" in data
    assert "swagger_ui_url" in data
    assert "redoc_url" in data

def test_predict_endpoint_valid_image(test_image):
    """Test the predict endpoint with a valid image"""
    response = client.post(
        "/predict",
        files={"file": ("test_image.png", test_image, "image/png")},
        params={"confidence": 0.5}
    )
    assert response.status_code == 200
    data = response.json()
    assert "threats" in data
    assert "recommendations" in data
    assert "details" in data
    
    # Verify the structure of threats if any are returned
    if data["threats"]:
        threat = data["threats"][0]
        assert "type" in threat
        assert "name" in threat
        assert "confidence" in threat
        assert 0.0 <= threat["confidence"] <= 1.0

def test_predict_endpoint_invalid_image_format():
    """Test the predict endpoint with an invalid image format"""
    # Create a text file instead of an image
    text_file = io.BytesIO(b"This is not an image")
    
    response = client.post(
        "/predict",
        files={"file": ("test.txt", text_file.getvalue(), "text/plain")},
        params={"confidence": 0.5}
    )
    assert response.status_code == 400
    data = response.json()
    assert "detail" in data
    assert "Invalid file type" in data["detail"]

def test_predict_endpoint_invalid_confidence():
    """Test the predict endpoint with invalid confidence values"""
    # Create a valid image
    image = Image.new('RGB', (100, 100), color='white')
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    
    # Test with confidence > 1.0
    response = client.post(
        "/predict",
        files={"file": ("test_image.png", img_byte_arr.getvalue(), "image/png")},
        params={"confidence": 1.5}
    )
    assert response.status_code == 422  # Validation error
    
    # Test with confidence < 0.0
    response = client.post(
        "/predict",
        files={"file": ("test_image.png", img_byte_arr.getvalue(), "image/png")},
        params={"confidence": -0.5}
    )
    assert response.status_code == 422  # Validation error

def test_predict_endpoint_confidence_filtering(test_image):
    """Test that confidence threshold correctly filters predictions"""
    # Test with low confidence threshold
    response_low = client.post(
        "/predict",
        files={"file": ("test_image.png", test_image, "image/png")},
        params={"confidence": 0.1}  # Very low threshold should include most predictions
    )
    assert response_low.status_code == 200
    threats_low = response_low.json()["threats"]
    
    # Test with high confidence threshold
    response_high = client.post(
        "/predict",
        files={"file": ("test_image.png", test_image, "image/png")},
        params={"confidence": 0.9}  # Very high threshold should exclude most predictions
    )
    assert response_high.status_code == 200
    threats_high = response_high.json()["threats"]
    
    # If we got predictions for both cases, high threshold should have fewer threats
    if threats_low and threats_high:
        assert len(threats_low) >= len(threats_high)

def test_predict_endpoint_missing_file():
    """Test the predict endpoint with missing file"""
    response = client.post("/predict", params={"confidence": 0.5})
    assert response.status_code == 422  # Validation error

if __name__ == "__main__":
    pytest.main(["-v"])
