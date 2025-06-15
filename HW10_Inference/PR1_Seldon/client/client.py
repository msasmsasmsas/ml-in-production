#!/usr/bin/env python3

"""
Client for making requests to the Seldon Core deployment.
"""

import argparse
import requests
import json
import numpy as np
from PIL import Image
import time
import sys
from typing import Dict, Any, Optional, List, Union

class SeldonClient:
    """Client for Seldon Core deployments."""

    def __init__(self, 
                 deployment_name: str = "resnet50-classifier",
                 namespace: str = "seldon",
                 host: str = "localhost",
                 port: int = 8003,
                 gateway: str = "ambassador"):
        """
        Initialize the Seldon client.

        Args:
            deployment_name: Name of the Seldon deployment
            namespace: Kubernetes namespace
            host: Host address
            port: Port number
            gateway: API gateway (ambassador or istio)
        """
        self.deployment_name = deployment_name
        self.namespace = namespace
        self.host = host
        self.port = port
        self.gateway = gateway

        # Construct base URL based on gateway type
        if self.gateway == "ambassador":
            self.base_url = f"http://{self.host}:{self.port}/seldon/{self.namespace}/{self.deployment_name}"
        elif self.gateway == "istio":
            self.base_url = f"http://{self.host}:{self.port}/seldon/{self.namespace}/{self.deployment_name}"
        else:
            raise ValueError(f"Unsupported gateway: {self.gateway}")

    def predict(self, 
                image_path: str, 
                model_name: str = "classifier") -> Dict[str, Any]:
        """
        Send prediction request to Seldon deployment.

        Args:
            image_path: Path to the image file
            model_name: Name of the model in the Seldon graph

        Returns:
            Prediction results
        """
        # Open and preprocess image
        image = Image.open(image_path).convert("RGB")
        image_data = self._preprocess_image(image)

        # Construct request payload
        payload = {
            "data": {
                "ndarray": image_data.tolist()
            }
        }

        # Construct request URL
        url = f"{self.base_url}/api/v1.0/predictions"

        # Send request
        start_time = time.time()
        response = requests.post(url, json=payload)
        response.raise_for_status()
        inference_time = time.time() - start_time

        # Parse response
        result = response.json()

        # Add inference time
        if "data" in result and isinstance(result["data"], dict):
            result["data"]["inference_time"] = round(inference_time, 4)

        return result

    def _preprocess_image(self, 
                          image: Image.Image, 
                          target_size: tuple = (224, 224)) -> np.ndarray:
        """
        Preprocess an image for the model.

        Args:
            image: PIL Image object
            target_size: Target image size (height, width)

        Returns:
            Preprocessed image as numpy array
        """
        # Resize image
        image = image.resize(target_size)

        # Convert to numpy array (0-255)
        img_array = np.array(image)

        # Return the raw image array - Seldon model will handle normalization
        return img_array

def main():
    """Main function for the client."""
    parser = argparse.ArgumentParser(description="Seldon Core Client")
    parser.add_argument("--image", type=str, required=True, help="Path to the image file")
    parser.add_argument("--host", type=str, default="localhost", help="Host address")
    parser.add_argument("--port", type=int, default=8003, help="Port number")
    parser.add_argument("--deployment", type=str, default="resnet50-classifier", help="Seldon deployment name")
    parser.add_argument("--namespace", type=str, default="seldon", help="Kubernetes namespace")
    parser.add_argument("--gateway", type=str, default="ambassador", choices=["ambassador", "istio"], 
                       help="API gateway type")
    args = parser.parse_args()

    # Create client
    client = SeldonClient(
        deployment_name=args.deployment,
        namespace=args.namespace,
        host=args.host,
        port=args.port,
        gateway=args.gateway
    )

    try:
        # Send prediction request
        print(f"Sending prediction request for image: {args.image}")
        result = client.predict(args.image)

        # Extract predictions
        predictions = None
        if "data" in result and isinstance(result["data"], dict):
            if "predictions" in result["data"]:
                predictions = result["data"]["predictions"]
            elif "ndarray" in result["data"] and isinstance(result["data"]["ndarray"], list):
                predictions = result["data"]["ndarray"]

        # Print results
        if predictions:
            print("\nPrediction results:")
            if isinstance(predictions, list) and all(isinstance(p, dict) for p in predictions):
                # If predictions are structured objects
                for pred in predictions:
                    print(f"{pred.get('rank', '-')}. {pred.get('class_name', 'Unknown')}: "
                         f"{pred.get('probability', 0.0):.4f}")
            else:
                # If predictions are raw values
                print(json.dumps(predictions, indent=2))

            # Print inference time if available
            if "inference_time" in result.get("data", {}):
                print(f"\nInference time: {result['data']['inference_time']}s")
        else:
            print("\nNo predictions found in response!")
            print("Raw response:")
            print(json.dumps(result, indent=2))

    except requests.exceptions.RequestException as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
