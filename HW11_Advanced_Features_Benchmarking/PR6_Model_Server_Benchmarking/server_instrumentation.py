import time
import json
import numpy as np
from flask import Flask, request, jsonify
import torch
from PIL import Image
import io

# Example of an instrumented REST server for benchmarking

app = Flask(__name__)

# Placeholder for a model - replace with your actual model
class DummyModel:
    def __init__(self):
        self.processing_time = 0.05  # Simulate model processing time

    def __call__(self, x):
        # Simulate model inference
        time.sleep(self.processing_time)
        return {"class_id": 42, "confidence": 0.95}

# Initialize model
model = DummyModel()

@app.route('/predict', methods=['POST'])
def predict():
    # Check if benchmarking is requested
    enable_timing = 'X-Benchmark' in request.headers

    timing_info = {}

    if enable_timing:
        start_time = time.time()

    # Preprocessing
    if enable_timing:
        preprocess_start = time.time()

    # Get image from request
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    image_file = request.files['image']
    image_bytes = image_file.read()

    # Simulate image preprocessing
    try:
        image = Image.open(io.BytesIO(image_bytes))
        # Convert to tensor, normalize, resize, etc.
        processed_image = np.array(image)  # Placeholder for actual preprocessing
    except Exception as e:
        return jsonify({'error': f'Image processing error: {str(e)}'}), 400

    if enable_timing:
        preprocess_end = time.time()
        timing_info['preprocessing'] = preprocess_end - preprocess_start

    # Inference
    if enable_timing:
        inference_start = time.time()

    # Run inference
    result = model(processed_image)

    if enable_timing:
        inference_end = time.time()
        timing_info['inference'] = inference_end - inference_start

    # Postprocessing
    if enable_timing:
        postprocess_start = time.time()

    # Process the model output
    processed_result = {
        "class_name": f"Class_{result['class_id']}",
        "confidence": result['confidence'],
        "prediction_time": time.time()
    }

    if enable_timing:
        postprocess_end = time.time()
        timing_info['postprocessing'] = postprocess_end - postprocess_start

    if enable_timing:
        end_time = time.time()
        timing_info['server_processing'] = end_time - start_time

    # Prepare response
    response = jsonify(processed_result)

    # Add timing information to response headers if benchmarking
    if enable_timing:
        response.headers['X-Timing-Info'] = json.dumps(timing_info)

    return response

# Example gRPC server would follow a similar pattern using the gRPC framework
# with timing information added to response metadata

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Run instrumented REST server')
    parser.add_argument('--port', type=int, default=5000, help='Server port')
    parser.add_argument('--host', default='0.0.0.0', help='Server host')
    args = parser.parse_args()

    print(f"Starting instrumented REST server on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False, threaded=True)
