<!-- Updated version for PR -->
# Model Server Benchmarking Toolkit

This toolkit provides comprehensive tools for benchmarking machine learning model servers, including REST and gRPC interfaces, and component-level performance analysis.

## Features

- **Comprehensive Benchmarking**: Measure latency, throughput, and resource utilization
- **Protocol Comparison**: Compare REST and gRPC performance
- **Component Analysis**: Identify bottlenecks in individual components
- **Visualization**: Generate insightful performance charts
- **Batch Size Optimization**: Find optimal batch sizes for your model

## Tools Included

### 1. General Benchmarking Tool (`benchmark.py`)

A versatile tool for benchmarking model servers with support for both REST and gRPC protocols.

```bash
python benchmark.py --image test_image.jpg --rest-url http://localhost:5000 --grpc-server localhost:50051 --requests 1000 --concurrency 20 --output-json results.json --output-plot results.png
```

### 2. Component Benchmarking Tool (`component_benchmark.py`)

A tool for analyzing the performance of individual components in the inference pipeline:

- Image loading
- Preprocessing
- Model forward pass
- Postprocessing
- End-to-end performance

```bash
python component_benchmark.py --image test_image.jpg --device cuda --iterations 100 --component all --output-json component_results.json --output-plot component_results.png
```

### 3. REST vs gRPC Comparison Tool (`rest_vs_grpc_benchmark.py`)

A specialized tool for comparing REST and gRPC performance with detailed analysis.

```bash
python rest_vs_grpc_benchmark.py --image test_image.jpg --rest-url http://localhost:5000 --grpc-server localhost:50051 --requests 500 --concurrency 10 --output-json protocol_results.json --output-plot protocol_results.png
```

## Performance Metrics

The toolkit collects and reports a wide range of metrics:

- **Latency**: min, max, mean, median, p90, p95, p99
- **Throughput**: requests per second (RPS)
- **Success Rate**: percentage of successful requests
- **Component Timing**: breakdown of time spent in each component
- **Resource Utilization**: CPU, memory, and GPU utilization (when available)

## Visualization

Results are visualized through various charts:

- Bar charts for latency and throughput comparisons
- Box plots for latency distribution
- Pie charts for component time contribution
- Line charts for concurrency scaling
- Histograms for request time distribution

## Requirements

- Python 3.7+
- PyTorch
- NumPy
- Matplotlib
- Pandas
- Seaborn (for enhanced visualizations)
- Requests (for REST benchmarking)
- gRPC tools (for gRPC benchmarking)

## Best Practices

- Run benchmarks on isolated environments to avoid interference
- Perform multiple runs and average the results for more reliable measurements
- Test with various concurrency levels to find optimal configuration
- Compare different batch sizes to maximize throughput
- Use realistic input data that matches your production workload

