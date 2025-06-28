#!/usr/bin/env python3

"""
Client for making requests to the Ray Serve deployment.
"""

import argparse
import requests
import json
from PIL import Image
import time
from typing import Dict, Any, Optional
import sys

class RayServeClient:
    """
    Client for the Ray Serve ML API.
    """

    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialize the client.

        Args:
            base_url: Base URL of the Ray Serve deployment
        """
        self.base_url = base_url.rstrip('/')

    def health_check(self) -> Dict[str, Any]:
        """
        Check if the service is healthy.

        Returns:
            Health status
        """
        response = requests.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()

    def get_metadata(self) -> Dict[str, Any]:
        """
        Get service metadata.

        Returns:
            Service metadata
        """
        response = requests.get(f"{self.base_url}/metadata")
        response.raise_for_status()
        return response.json()

    def predict(self, image_path: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Send prediction request.

        Args:
            image_path: Path to the image file
            top_k: Number of top predictions to return

        Returns:
            Prediction results
        """
        # Open the image file
        with open(image_path, "rb") as f:
            files = {"file": (image_path, f, "image/jpeg")}
            params = {"top_k": top_k}

            # Send the request
            response = requests.post(
                f"{self.base_url}/predict",
                files=files,
                params=params
            )

        # Check for errors
        response.raise_for_status()

        # Return the results
        return response.json()

def main():
    parser = argparse.ArgumentParser(description="Ray Serve ML API Client")
    parser.add_argument(
        "--url", type=str, default="http://localhost:8000",
        help="URL of the Ray Serve deployment"
    )
    parser.add_argument(
        "--image", type=str, required=True,
        help="Path to the image file to classify"
    )
    parser.add_argument(
        "--top-k", type=int, default=5,
        help="Number of top predictions to return"
    )
    parser.add_argument(
        "--check-health", action="store_true",
        help="Check service health"
    )
    parser.add_argument(
        "--metadata", action="store_true",
        help="Get service metadata"
    )
    args = parser.parse_args()

    # Create client
    client = RayServeClient(args.url)

    try:
        # Check health if requested
        if args.check_health:
            health = client.health_check()
            print("Health check:", json.dumps(health, indent=2))

        # Get metadata if requested
        if args.metadata:
            metadata = client.get_metadata()
            print("Service metadata:", json.dumps(metadata, indent=2))

        # Classify image
        if args.image:
            print(f"\nClassifying image: {args.image}")
            start_time = time.time()
            results = client.predict(args.image, args.top_k)
            total_time = time.time() - start_time

            # Print results
            print(f"\nPrediction results (total time: {total_time:.4f}s):")

            # Перевірка наявності необхідних полів
            if 'model_name' in results and 'model_version' in results:
                print(f"Model: {results['model_name']} (version {results['model_version']})")

            if 'inference_time' in results:
                print(f"Inference time: {results['inference_time']}s")

            print("\nTop predictions:")

            if 'predictions' in results:
                for i, pred in enumerate(results['predictions']):
                    print(f"{i+1}. {pred['class_name']}: {pred['probability']:.4f}")
            else:
                print("No predictions found in response")
                print(f"Response content: {results}")

    except requests.exceptions.RequestException as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
