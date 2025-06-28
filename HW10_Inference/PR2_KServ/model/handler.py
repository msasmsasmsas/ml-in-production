#!/usr/bin/env python3

"""
Custom handler for ResNet50 model in TorchServe/KServe.
"""

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from ts.torch_handler.base_handler import BaseHandler
from PIL import Image
from typing import List, Dict, Any, Union
import io
import json
import os
import logging
import time

from model import ResNet50Model

logger = logging.getLogger(__name__)

class ResNet50Handler(BaseHandler):
    """Custom handler for ResNet50 model."""

    def __init__(self):
        """
        Initialize the handler.
        """
        super().__init__()
        self.initialized = False
        self.model = None
        self.device = None
        self.transform = None
        self.class_to_idx = None
        self.topk = 5
        self.explain_mode = False

    def initialize(self, context):
        """
        Initialize the handler with model artifacts.

        Args:
            context: TorchServe context
        """
        logger.info("Initializing ResNet50 handler")

        # Get properties from context
        self.manifest = context.manifest
        properties = context.system_properties
        model_dir = properties.get("model_dir")
        self.device = torch.device("cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu")

        # Initialize the model
        self.model = ResNet50Model()

        # Define image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        self.initialized = True
        logger.info(f"ResNet50 handler initialized successfully. Using device: {self.device}")

    def preprocess(self, requests):
        """
        Preprocess input data.

        Args:
            requests: List of inference requests

        Returns:
            Preprocessed tensor batch
        """
        # Process each request
        images = []

        for request in requests:
            # Get request parameters
            if self._is_explain_request(request):
                self.explain_mode = True
            else:
                self.explain_mode = False

            # Get topk parameter if specified
            self.topk = self._get_topk_param(request)

            # Get image data
            image_data = request.get("data") or request.get("body")

            # Handle different input formats
            image = self._convert_to_image(image_data)

            # Apply transformations
            image_tensor = self.transform(image)
            images.append(image_tensor)

        # Create batch tensor
        batch = torch.stack(images).to(self.device)

        return batch

    def inference(self, batch):
        """
        Run inference on the preprocessed batch.

        Args:
            batch: Preprocessed tensor batch

        Returns:
            Model predictions
        """
        # Start timing
        start_time = time.time()

        # Run inference
        if self.explain_mode:
            results = self.model.explain(batch)
        else:
            predictions, _ = self.model.predict(batch)
            results = predictions

        # Record inference time
        inference_time = time.time() - start_time
        logger.info(f"Inference time: {inference_time:.4f} seconds")

        return results

    def postprocess(self, inference_output):
        """
        Postprocess the model output.

        Args:
            inference_output: Model output from inference

        Returns:
            Formatted response
        """
        # Format response based on explain mode
        if self.explain_mode:
            # Return explanation results directly
            responses = [inference_output]
        else:
            # Format predictions for each item in the batch
            responses = []
            for predictions in inference_output:
                # Limit to topk predictions
                predictions = predictions[:self.topk]

                # Format response
                response = {
                    "predictions": predictions,
                    "model_name": self.model.name,
                    "model_version": self.model.version,
                    "inference_time": time.time() - self.start_time if hasattr(self, 'start_time') else None
                }
                responses.append(response)

        return responses

    def handle(self, data, context):
        """
        Handle the inference request.

        Args:
            data: Input data
            context: TorchServe context

        Returns:
            Inference response
        """
        # Store start time for timing
        self.start_time = time.time()

        try:
            # Call parent handle method
            response = super().handle(data, context)

            # Add overall processing time
            if isinstance(response, list) and len(response) > 0 and isinstance(response[0], dict):
                for resp in response:
                    resp["total_time"] = time.time() - self.start_time

            return response

        except Exception as e:
            logger.error(f"Error handling request: {e}", exc_info=True)
            return [{
                "error": str(e),
                "status": "failed"
            }]

    def _convert_to_image(self, data):
        """
        Convert input data to PIL Image.

        Args:
            data: Input data (bytes, JSON, etc.)

        Returns:
            PIL Image
        """
        # If data is bytes, assume it's a raw image
        if isinstance(data, bytes):
            return Image.open(io.BytesIO(data)).convert("RGB")

        # If data is a string, try to load it as a file path or base64
        if isinstance(data, str):
            try:
                # Try to parse as JSON
                data = json.loads(data)
            except json.JSONDecodeError:
                # If not JSON, assume it's a file path
                if os.path.isfile(data):
                    return Image.open(data).convert("RGB")
                # Otherwise, try to decode as base64
                import base64
                try:
                    image_bytes = base64.b64decode(data)
                    return Image.open(io.BytesIO(image_bytes)).convert("RGB")
                except Exception:
                    raise ValueError(f"Unsupported string input: {data[:100]}...")

        # Handle dict input (common in API requests)
        if isinstance(data, dict):
            # Check for base64 encoded image
            if "b64" in data:
                import base64
                image_bytes = base64.b64decode(data["b64"])
                return Image.open(io.BytesIO(image_bytes)).convert("RGB")

            # Check for URL
            if "url" in data:
                import requests
                response = requests.get(data["url"], stream=True)
                response.raise_for_status()
                return Image.open(io.BytesIO(response.content)).convert("RGB")

            # Check for file path
            if "path" in data and os.path.isfile(data["path"]):
                return Image.open(data["path"]).convert("RGB")

        # If we get here, we couldn't convert the input
        raise ValueError(f"Unsupported input type: {type(data)}")

    def _is_explain_request(self, request):
        """
        Check if the request is an explanation request.

        Args:
            request: Request data

        Returns:
            Boolean indicating if it's an explain request
        """
        # Check explicit explain flag
        if isinstance(request, dict) and "explain" in request:
            return request["explain"] is True

        # Check in metadata
        if isinstance(request, dict) and "metadata" in request:
            metadata = request["metadata"]
            if isinstance(metadata, dict) and "explain" in metadata:
                return metadata["explain"] is True

        return False

    def _get_topk_param(self, request):
        """
        Extract topk parameter from request.

        Args:
            request: Request data

        Returns:
            topk value (default is 5)
        """
        # Default value
        default_topk = 5

        # Check direct parameter
        if isinstance(request, dict) and "topk" in request:
            try:
                return int(request["topk"])
            except (ValueError, TypeError):
                return default_topk

        # Check in parameters or metadata
        for key in ["parameters", "metadata"]:
            if isinstance(request, dict) and key in request and isinstance(request[key], dict):
                if "topk" in request[key]:
                    try:
                        return int(request[key]["topk"])
                    except (ValueError, TypeError):
                        return default_topk

        return default_topk
