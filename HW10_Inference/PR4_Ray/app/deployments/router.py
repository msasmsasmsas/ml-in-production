#!/usr/bin/env python3

"""
Router deployment for handling HTTP requests in Ray Serve.
"""

from ray import serve
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse
from typing import Dict, Any, List
import numpy as np
from PIL import Image
import io
import time

class RouterDeployment:
    def __init__(self, classifier):
        """Initialize the router with a reference to the classifier deployment."""
        self.classifier = classifier
        self.app = FastAPI()

        # Set up routes
        self.app.post("/predict")(self.predict)
        self.app.get("/health")(self.health)
        self.app.get("/metadata")(self.metadata)

    async def predict(self, 
                      file: UploadFile = File(...), 
                      top_k: int = Query(5, ge=1, le=100)):
        """
        Endpoint for image classification.

        Args:
            file: Uploaded image file
            top_k: Number of top predictions to return

        Returns:
            Prediction results as JSON
        """
        # Check file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")

        try:
            # Read and preprocess the image
            start_time = time.time()
            image_bytes = await file.read()
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

            # Preprocess the image
            processed_image = self._preprocess_image(image)

            # Get predictions from classifier
            results = await self.classifier.remote(processed_image, top_k)

            # Add processing time
            inference_time = time.time() - start_time
            results["inference_time"] = round(inference_time, 4)

            return JSONResponse(content=results)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

    async def health(self):
        """Health check endpoint."""
        return {"status": "healthy"}

    async def metadata(self):
        """Endpoint to get model metadata."""
        return {
            "service": "Ray Serve Image Classification API",
            "version": "1.0.0",
            "frameworks": ["Ray", "PyTorch", "FastAPI"],
            "endpoints": [
                {
                    "path": "/predict",
                    "method": "POST",
                    "description": "Classify an image"
                },
                {
                    "path": "/health",
                    "method": "GET",
                    "description": "Health check"
                },
                {
                    "path": "/metadata",
                    "method": "GET",
                    "description": "Service metadata"
                }
            ]
        }

    def _preprocess_image(self, image, target_size=(224, 224)):
        """
        Preprocess an image for the model.

        Args:
            image: PIL Image object
            target_size: Target size for resizing

        Returns:
            Preprocessed image as numpy array
        """
        # Resize image
        image = image.resize(target_size)

        # Convert to numpy array and normalize
        img_array = np.array(image).astype(np.float32) / 255.0

        # Apply normalization for pre-trained models
        mean = np.array([0.485, 0.456, 0.406]).reshape((1, 1, 3))
        std = np.array([0.229, 0.224, 0.225]).reshape((1, 1, 3))
        img_array = (img_array - mean) / std

        # Transpose from (H, W, C) to (C, H, W)
        img_array = img_array.transpose(2, 0, 1)

        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)

        return img_array

    async def __call__(self, request):
        """Handle HTTP requests.

        Ray Serve викликає цей метод для обробки HTTP-запитів.
        """
        try:
            # Обробка запитів на основі URL-шляху та HTTP-методу
            if request.method == "GET":
                if request.url.path == "/health":
                    return await self.health()
                elif request.url.path == "/metadata":
                    return await self.metadata()
                else:
                    return {"error": "Endpoint not found", "status_code": 404}
            elif request.method == "POST" and request.url.path == "/predict":
                # Отримуємо файл з multipart/form-data запиту
                form = await request.form()
                file = form.get("file")

                if not file:
                    return {"error": "No file provided", "status_code": 400}

                # Отримуємо параметр top_k з query параметрів
                top_k = int(request.query_params.get("top_k", 5))

                # Викликаємо метод predict
                try:
                    # Зчитуємо вміст файлу
                    content = await file.read()
                    image = Image.open(io.BytesIO(content)).convert('RGB')

                    # Обробляємо зображення
                    processed_image = self._preprocess_image(image)

                    # Отримуємо прогнози від класифікатора
                    start_time = time.time()
                    results = await self.classifier.remote(processed_image, top_k)

                    # Додаємо час обробки
                    inference_time = time.time() - start_time
                    results["inference_time"] = round(inference_time, 4)

                    return results
                except Exception as e:
                    print(f"Error processing image: {str(e)}")
                    return {"error": f"Error processing image: {str(e)}", "status_code": 500}
            else:
                return {"error": "Method not allowed", "status_code": 405}
        except Exception as e:
            print(f"Unexpected error in __call__: {str(e)}")
            return {"error": f"Server error: {str(e)}", "status_code": 500}
