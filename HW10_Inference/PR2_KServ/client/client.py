#!/usr/bin/env python3

"""
Client for sending inference requests to KServe.
"""

import argparse
import requests
import json
import base64
import time
import sys
from typing import Dict, Any, Optional, List, Union
from PIL import Image
import os

class KServeClient:
    """Client for KServe inference service."""

    def __init__(self, 
                 service_name: str = "resnet50-classifier",
                 namespace: str = "kserve-demo",
                 host: str = "localhost",
                 port: int = 8080):
        """
        Initialize the KServe client.

        Args:
            service_name: Name of the InferenceService
            namespace: Kubernetes namespace
            host: Host address
            port: Port number
        """
        self.service_name = service_name
        self.namespace = namespace
        self.host = host
        self.port = port

        # Construct base URL
        self.base_url = f"http://{self.host}:{self.port}/v1/models/{self.service_name}"

    def predict(self, image_path: str, topk: int = 5) -> Dict[str, Any]:
        """
        Send prediction request to KServe.

        Args:
            image_path: Path to the image file
            topk: Number of top predictions to return

        Returns:
            Prediction results
        """
        # Open and encode image
        with open(image_path, "rb") as f:
            image_bytes = f.read()

        # Encode image to base64
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")

        # Construct request payload
        payload = {
            "instances": [
                {
                    "data": {
                        "b64": image_b64
                    },
                    "parameters": {
                        "topk": topk
                    }
                }
            ]
        }

        # Construct request URL
        url = f"{self.base_url}/infer"

        # Send request
        start_time = time.time()
        response = requests.post(url, json=payload)
        response.raise_for_status()
        total_time = time.time() - start_time

        # Parse response
        result = response.json()

        # Add timing information
        if "predictions" in result:
            result["timing"] = {
                "total_time": round(total_time, 4)
            }

        return result

    def explain(self, image_path: str) -> Dict[str, Any]:
        """
        Send explanation request to KServe.

        Args:
            image_path: Path to the image file

        Returns:
            Explanation results
        """
        # Open and encode image
        with open(image_path, "rb") as f:
            image_bytes = f.read()

        # Encode image to base64
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")

        # Construct request payload
        payload = {
            "instances": [
                {
                    "data": {
                        "b64": image_b64
                    },
                    "metadata": {
                        "explain": True
                    }
                }
            ]
        }

        # Construct request URL
        url = f"{self.base_url}/explain"

        # Send request
        start_time = time.time()
        response = requests.post(url, json=payload)
        response.raise_for_status()
        total_time = time.time() - start_time

        # Parse response
        result = response.json()

        # Add timing information
        result["timing"] = {
            "total_time": round(total_time, 4)
        }

        return result

    def health(self) -> Dict[str, Any]:
        """
        Check health status of the model.

        Returns:
            Health status
        """
        url = f"{self.base_url}/ready"
        response = requests.get(url)
        response.raise_for_status()
        return {"status": "healthy" if response.status_code == 200 else "unhealthy"}

    def metadata(self) -> Dict[str, Any]:
        """
        Get model metadata.

        Returns:
            Model metadata
        """
        url = f"{self.base_url}"
        response = requests.get(url)
        response.raise_for_status()
        return response.json()

def main():
    """Main function for the client."""
    parser = argparse.ArgumentParser(description="KServe Client")
    parser.add_argument("--image", type=str, required=True, help="Path to the image file")
    parser.add_argument("--host", type=str, default="localhost", help="Host address")
    parser.add_argument("--port", type=int, default=8080, help="Port number")
    parser.add_argument("--service", type=str, default="resnet50-classifier", help="InferenceService name")
    parser.add_argument("--namespace", type=str, default="kserve-demo", help="Kubernetes namespace")
    parser.add_argument("--explain", action="store_true", help="Request explanation")
    parser.add_argument("--topk", type=int, default=5, help="Number of top predictions to return")
    parser.add_argument("--health", action="store_true", help="Check health status")
    parser.add_argument("--metadata", action="store_true", help="Get model metadata")
    args = parser.parse_args()

    # Create client
    client = KServeClient(
        service_name=args.service,
        namespace=args.namespace,
        host=args.host,
        port=args.port
    )

    try:
        # Check health if requested
        if args.health:
            health = client.health()
            print("Health status:", json.dumps(health, indent=2))

        # Get metadata if requested
        if args.metadata:
            metadata = client.metadata()
            print("Model metadata:", json.dumps(metadata, indent=2))

        # Process image if provided
        if args.image:
            if not os.path.exists(args.image):
                print(f"Error: Image file not found: {args.image}", file=sys.stderr)
                sys.exit(1)

            print(f"Processing image: {args.image}")

            # Get explanation or prediction
            if args.explain:
                result = client.explain(args.image)
                print("\nExplanation results:")
                print(json.dumps(result, indent=2))
            else:
                result = client.predict(args.image, args.topk)

                # Print prediction results
                if "predictions" in result and len(result["predictions"]) > 0:
                    predictions = result["predictions"][0]

                    print("\nPrediction results:")
                    if isinstance(predictions, list):
                        for pred in predictions:
                            if isinstance(pred, dict) and "class_name" in pred and "probability" in pred:
                                print(f"{pred.get('rank', '-')}. {pred['class_name']}: {pred['probability']:.4f}")
                            else:
                                print(pred)
                    else:
                        print(json.dumps(predictions, indent=2))

                    # Print timing information
                    if "timing" in result:
                        print(f"\nTotal time: {result['timing']['total_time']}s")
                else:
                    print("\nRaw response:")
                    print(json.dumps(result, indent=2))

    except requests.exceptions.RequestException as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
