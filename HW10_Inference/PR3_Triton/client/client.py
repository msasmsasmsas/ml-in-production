#!/usr/bin/env python3

"""
Client code for sending inference requests to Triton Inference Server.
"""

import argparse
import numpy as np
import tritonclient.http as httpclient
from PIL import Image
import json
import os
import sys
from tritonclient.utils import triton_to_np_dtype

# ImageNet class labels
try:
    with open('client/imagenet_classes.json', 'r') as f:
        IMAGENET_CLASSES = json.load(f)
except (FileNotFoundError, json.JSONDecodeError):
    # Базові класи ImageNet для демонстрації
    IMAGENET_CLASSES = {
        "0": "tench", "1": "goldfish", "2": "great white shark", "3": "tiger shark",
        "4": "hammerhead", "5": "electric ray", "6": "stingray", "7": "cock",
        "8": "hen", "9": "ostrich", "10": "brambling"
    }

def preprocess_image(image_path, target_size=(224, 224)):
    """
    Preprocess an image for the model.

    Args:
        image_path: Path to the image file
        target_size: Target size for resizing

    Returns:
        Preprocessed image as numpy array
    """
    try:
        # Load and resize image
        img = Image.open(image_path).convert('RGB')
        img = img.resize(target_size)

        # Convert to numpy array and normalize - явно вказуємо тип float32
        img_array = np.array(img, dtype=np.float32) / 255.0

        # Apply normalization for pre-trained models - явно вказуємо тип float32
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape((1, 1, 3))
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape((1, 1, 3))
        img_array = (img_array - mean) / std

        # Transpose from (H, W, C) to (C, H, W)
        img_array = img_array.transpose(2, 0, 1)

        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)

        # Перевірка та примусове приведення до float32
        if img_array.dtype != np.float32:
            print(f"Увага: приведення типу даних з {img_array.dtype} до float32")
            img_array = img_array.astype(np.float32)

        return img_array

    except Exception as e:
        print(f"Помилка при обробці зображення: {e}")
        raise

def infer(image_path, model_name, url='localhost:8000'):
    """
    Send inference request to Triton server.

    Args:
        image_path: Path to the image file
        model_name: Name of the model on Triton server
        url: Triton server URL

    Returns:
        Prediction result
    """
    # Create a client
    client = httpclient.InferenceServerClient(url=url)

    # Check if the model is ready
    if not client.is_model_ready(model_name):
        raise RuntimeError(f"Model {model_name} is not ready")

    # Get model metadata
    model_metadata = client.get_model_metadata(model_name)
    model_config = client.get_model_config(model_name)

    # Preprocess the image
    input_data = preprocess_image(image_path)

    # Create input tensor
    inputs = []
    inputs.append(httpclient.InferInput('input', input_data.shape, "FP32"))
    inputs[0].set_data_from_numpy(input_data)

    # Create output tensor
    outputs = []
    outputs.append(httpclient.InferRequestedOutput('output'))

    # Send the inference request
    results = client.infer(model_name, inputs, outputs=outputs)

    # Get the output data
    output_data = results.as_numpy('output')

    # Process the output (get top 5 predictions)
    topk = 5
    indices = np.argsort(output_data[0])[-topk:][::-1]
    probabilities = output_data[0][indices].astype(np.float32)

    # Format the results
    predictions = []
    for i, idx in enumerate(indices):
        idx_str = str(idx)
        class_name = IMAGENET_CLASSES.get(idx_str, f"Unknown class {idx_str}")
        predictions.append({
            'class': class_name,
            'probability': float(probabilities[i].astype(np.float32))
        })

    return predictions

def main():
    parser = argparse.ArgumentParser(description='Triton Inference Client')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--model', type=str, default='resnet50', help='Model name on Triton server')
    parser.add_argument('--url', type=str, default='localhost:8000', help='Triton server URL')
    parser.add_argument('--verbose', action='store_true', help='Включити детальне логування')

    args = parser.parse_args()

    print(f"Запуск інференсу для зображення: {args.image}")
    print(f"Використовуємо модель: {args.model}")
    print(f"URL сервера: {args.url}")

    if not os.path.exists(args.image):
        print(f"Помилка: файл зображення '{args.image}' не знайдено")
        sys.exit(1)

    try:
        results = infer(args.image, args.model, args.url)
        print("\nТоп-5 прогнозів:")
        for i, prediction in enumerate(results):
            print(f"{i+1}. {prediction['class']}: {prediction['probability']:.4f}")
    except Exception as e:
        print(f"Помилка під час інференсу: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
