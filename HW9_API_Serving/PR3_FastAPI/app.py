from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import numpy as np
from PIL import Image
import io
import torch
from pathlib import Path
import logging
import time
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("api.log")
    ]
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Agricultural Threat Detection API",
    description="API for detecting threats to agricultural crops from images",
    version="1.0.0",
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define response models
class Threat(BaseModel):
    type: str = Field(..., description="Type of threat (disease, pest, weed)")
    name: str = Field(..., description="Name of the specific threat")
    confidence: float = Field(..., description="Confidence score between 0 and 1", ge=0.0, le=1.0)

class PredictionResponse(BaseModel):
    threats: List[Threat] = Field(default_factory=list, description="List of detected threats")
    recommendations: List[str] = Field(default_factory=list, description="List of recommendations")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Additional details about the analysis")

# Mock model class - in production, replace with actual model
class ThreatDetectionModel:
    def __init__(self, model_path: str = "models/threat_detection_model.pt"):
        # In a real implementation, load the model here
        self.model_path = model_path
        logger.info(f"Initializing model from {model_path}")
        # Mock model initialization
        # self.model = torch.load(model_path) if os.path.exists(model_path) else None
        self.model = None
        self.class_names = [
            "healthy", "late_blight", "early_blight", "rust", 
            "septoria", "aphids", "thrips", "whitefly", 
            "bindweed", "nutsedge", "chickweed"
        ]
        logger.info("Model initialized successfully")
    
    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        # Implement proper preprocessing for your model
        logger.debug("Preprocessing image")
        img_array = np.array(image.convert("RGB").resize((224, 224)))
        # Convert to tensor, normalize, etc.
        # Example preprocessing (replace with actual preprocessing)
        tensor = torch.tensor(img_array).float().permute(2, 0, 1) / 255.0
        return tensor.unsqueeze(0)
    
    def predict(self, image: Image.Image, confidence_threshold: float = 0.5) -> Dict[str, Any]:
        """
        Process image and return predictions
        """
        logger.info(f"Running prediction with confidence threshold: {confidence_threshold}")
        start_time = time.time()
        
        # Preprocess image
        tensor = self.preprocess_image(image)
        
        # In a real implementation, run model inference
        # predictions = self.model(tensor)
        
        # Mock predictions for testing
        # Random predictions for demonstration purposes
        np.random.seed(int(tensor.sum().item() * 100))  # Make predictions deterministic based on image content
        
        mock_predictions = []
        # Generate 0-3 random threats
        num_threats = np.random.randint(0, 4)
        
        for _ in range(num_threats):
            threat_type = np.random.choice(["disease", "pest", "weed"])
            confidence = np.random.uniform(0.3, 0.95)
            
            # Skip if below threshold
            if confidence < confidence_threshold:
                continue
                
            # Select name based on type
            if threat_type == "disease":
                name = np.random.choice(["late_blight", "early_blight", "rust", "septoria"])
            elif threat_type == "pest":
                name = np.random.choice(["aphids", "thrips", "whitefly"])
            else:  # weed
                name = np.random.choice(["bindweed", "nutsedge", "chickweed"])
                
            mock_predictions.append({
                "type": threat_type,
                "name": name,
                "confidence": float(confidence)  # Ensure it's a native Python float
            })
        
        # Generate recommendations based on detected threats
        recommendations = []
        details = {}
        
        if not mock_predictions:
            recommendations.append("No threats detected. Continue regular crop monitoring.")
        else:
            for pred in mock_predictions:
                if pred["type"] == "disease":
                    recommendations.append(f"For {pred['name']}, consider applying appropriate fungicide.")
                elif pred["type"] == "pest":
                    recommendations.append(f"For {pred['name']}, consider using insecticide or biological control.")
                else:  # weed
                    recommendations.append(f"For {pred['name']}, consider mechanical removal or targeted herbicide.")
            
            # Add some details
            details = {
                "severity": "low" if len(mock_predictions) == 1 else "moderate" if len(mock_predictions) == 2 else "high",
                "processing_time_ms": round((time.time() - start_time) * 1000),
                "image_quality": "good",
                "model_version": "1.0.0"
            }
        
        logger.info(f"Prediction completed in {time.time() - start_time:.2f} seconds")
        return {
            "threats": mock_predictions,
            "recommendations": recommendations,
            "details": details
        }

# Initialize model
model = ThreatDetectionModel()

@app.get("/")
async def root():
    """
    Root endpoint - health check
    """
    return {
        "status": "online",
        "message": "Agricultural Threat Detection API is running",
        "version": "1.0.0"
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(...),
    confidence: float = Query(0.5, ge=0.0, le=1.0, description="Confidence threshold for detections")
):
    """
    Process uploaded image and return prediction results
    """
    logger.info(f"Received prediction request with confidence threshold: {confidence}")
    
    # Validate file type
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        logger.warning(f"Invalid file type: {file.content_type}")
        raise HTTPException(status_code=400, detail="Invalid file type. Only JPEG and PNG images are supported.")
    
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Make prediction
        result = model.predict(image, confidence)
        
        logger.info(f"Prediction successful. Found {len(result['threats'])} threats.")
        return result
    
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.get("/health")
async def health_check():
    """
    Health check endpoint for monitoring
    """
    return {
        "status": "healthy",
        "time": time.time(),
        "model_loaded": model is not None
    }

@app.get("/docs-info")
async def docs_info():
    """
    Provide information about the API documentation
    """
    return {
        "openapi_url": "/openapi.json",
        "swagger_ui_url": "/docs",
        "redoc_url": "/redoc"
    }

if __name__ == "__main__":
    # Run the FastAPI app using Uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
