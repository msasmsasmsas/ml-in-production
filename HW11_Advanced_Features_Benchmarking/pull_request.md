# Pull Request: Advanced Features & Benchmarking

This PR implements the advanced features and benchmarking tools for the model serving system, including:

## PR1: Dynamic Request Batching
Implements dynamic batching of inference requests to improve throughput under high load.

## PR2: Ensemble of Several Models
Support for running an ensemble of multiple models and aggregating their predictions.

## PR3: gRPC Inference
Implements a high-performance gRPC interface for model inference.

## PR4: Benchmarking
Basic benchmarking tools for measuring inference performance.

## PR5: REST vs gRPC Benchmark
Comprehensive tools for comparing REST and gRPC performance characteristics.

## PR6: Model Server Benchmarking by Components
Detailed benchmarking tools that break down the performance of model server components:
- Data preprocessing time
- Model inference (forward pass) time
- Postprocessing time
- Network latency
- Total server processing time

### Key Features of PR6
- Support for both REST and gRPC protocols
- Component-level timing measurements
- Statistical analysis of performance metrics
- Visualization tools for performance comparison
- Concurrent request benchmarking

### How PR6 Works
- Instruments servers to measure component timings
- Sends benchmark requests with timing flags enabled
- Collects and analyzes timing data from responses
- Generates comprehensive visualizations

### Testing
The benchmarking tools have been tested with various model sizes and under different load conditions. All modules pass the provided test suite.

### Documentation
Each module includes detailed documentation on usage, configuration, and result interpretation.
