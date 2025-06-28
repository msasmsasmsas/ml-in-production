# REST vs gRPC Benchmark

This module provides comprehensive benchmarking tools for comparing REST and gRPC performance for machine learning model serving.

## Features

- Direct comparison of REST and gRPC protocols
- Performance measurement under various load conditions
- Detailed latency and throughput metrics
- Visualizations for performance comparison
- Support for concurrent request testing

## Components

### 1. REST Server

A Flask-based REST API server for model inference.

### 2. gRPC Server

A high-performance gRPC server for model inference.

### 3. Benchmark Client

A versatile client for benchmarking both REST and gRPC endpoints.

### 4. Result Analyzer

Tools for analyzing and visualizing benchmark results.

## Getting Started

### Starting the Servers

```bash
# Start the REST server
python rest_server.py --port 5000

# Start the gRPC server
python grpc_server.py --port 50051
```

### Running the Benchmark

```bash
python benchmark.py --image test_image.jpg --rest-url http://localhost:5000 --grpc-server localhost:50051 --requests 1000 --concurrency 10 --output-json results.json --output-plot results.png
```

## Understanding the Results

The benchmarking tools provide several key metrics:

### Latency Metrics

- **Average Latency**: The mean time for a request to complete
- **Percentile Latencies**: p50, p90, p95, p99 latencies
- **Minimum/Maximum Latency**: The best and worst-case latencies

### Throughput Metrics

- **Requests Per Second (RPS)**: The number of requests processed per second
- **Maximum Throughput**: The peak throughput achieved
- **Throughput vs Concurrency**: How throughput scales with concurrent users

### Other Metrics

- **Error Rate**: Percentage of failed requests
- **Network Overhead**: Difference between client and server processing times
- **Time-to-First-Byte**: How quickly the server begins responding

## Interpreting the Visualizations

- **Bar Charts**: Compare key metrics between REST and gRPC
- **Line Charts**: Show how metrics change with increasing load
- **Box Plots**: Display the distribution of request latencies
- **Histograms**: Reveal the frequency distribution of latencies

## Key Findings

Typical benchmark results show:

1. **Latency**: gRPC generally provides lower latency than REST, especially for small payloads
2. **Throughput**: gRPC achieves higher throughput, particularly under high concurrency
3. **Resource Usage**: gRPC typically uses less CPU and memory per request
4. **Scalability**: gRPC scales better with increasing concurrent connections

## When to Choose Each Protocol

### REST Advantages

- Universal compatibility and familiarity
- Simpler implementation and debugging
- Better for browser-based clients
- Easier to test with standard tools

### gRPC Advantages

- Lower latency and higher throughput
- Built-in streaming capabilities
- Strongly typed contracts with Protocol Buffers
- Better support for bidirectional communication

## Best Practices

- Use gRPC for internal service-to-service communication
- Consider REST for public APIs and browser clients
- For high-volume inference, gRPC offers significant performance benefits
- Implement both protocols with a shared backend for flexibility
