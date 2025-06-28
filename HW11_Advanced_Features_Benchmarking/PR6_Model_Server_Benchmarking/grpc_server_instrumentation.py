import time
import argparse
import grpc
import numpy as np
from PIL import Image
import io
from concurrent import futures

# Import the generated protobuf code
# In a real implementation, these would be generated from the .proto file
try:
    import inference_pb2
    import inference_pb2_grpc
except ImportError:
    print("Error: Proto modules not found. Run 'python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. inference.proto' first.")
    exit(1)

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

class InferenceServicer(inference_pb2_grpc.InferenceServiceServicer):
    """Implementation of gRPC inference service with timing instrumentation"""

    def Predict(self, request, context):
        """Handle single image prediction request with timing information"""
        # Check if benchmarking is requested
        enable_timing = request.enable_timing

        timing_info = []

        if enable_timing:
            start_time = time.time()

        # Preprocessing
        if enable_timing:
            preprocess_start = time.time()

        # Get image from request
        image_bytes = request.image_data

        # Simulate image preprocessing
        try:
            image = Image.open(io.BytesIO(image_bytes))
            # Convert to tensor, normalize, resize, etc.
            processed_image = np.array(image)  # Placeholder for actual preprocessing
        except Exception as e:
            return inference_pb2.PredictResponse(error=f"Image processing error: {str(e)}")

        if enable_timing:
            preprocess_end = time.time()
            timing_info.append(inference_pb2.TimingInfo(
                component="preprocessing",
                duration_ms=(preprocess_end - preprocess_start) * 1000
            ))

        # Inference
        if enable_timing:
            inference_start = time.time()

        # Run inference
        result = model(processed_image)

        if enable_timing:
            inference_end = time.time()
            timing_info.append(inference_pb2.TimingInfo(
                component="inference",
                duration_ms=(inference_end - inference_start) * 1000
            ))

        # Postprocessing
        if enable_timing:
            postprocess_start = time.time()

        # Process the model output
        import json
        processed_result = {
            "class_name": f"Class_{result['class_id']}",
            "confidence": result['confidence'],
            "prediction_time": time.time()
        }
        result_json = json.dumps(processed_result)

        if enable_timing:
            postprocess_end = time.time()
            timing_info.append(inference_pb2.TimingInfo(
                component="postprocessing",
                duration_ms=(postprocess_end - postprocess_start) * 1000
            ))

        if enable_timing:
            end_time = time.time()
            timing_info.append(inference_pb2.TimingInfo(
                component="server_processing",
                duration_ms=(end_time - start_time) * 1000
            ))

        # Return response with timing information
        return inference_pb2.PredictResponse(
            result=result_json,
            timing_info=timing_info
        )

    def StreamingPredict(self, request_iterator, context):
        """Handle streaming prediction requests"""
        for request in request_iterator:
            # Process each request similar to Predict method
            response = self.Predict(request, context)
            yield response

def serve(port):
    """Start the gRPC server"""
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    inference_pb2_grpc.add_InferenceServiceServicer_to_server(
        InferenceServicer(), server)
    server.add_insecure_port(f'[::]:{port}')
    server.start()
    print(f"Started instrumented gRPC server on port {port}")
    server.wait_for_termination()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run instrumented gRPC server')
    parser.add_argument('--port', type=int, default=50051, help='Server port')
    args = parser.parse_args()

    serve(args.port)
