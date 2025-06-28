# gRPC Inference Server

This module implements a high-performance gRPC-based inference server for machine learning models.

## Features

- High-performance binary protocol (gRPC)
- Support for unary and streaming inference requests
- Health check endpoint
- Detailed performance metrics
- Support for large payloads

## Architecture

The system is built using gRPC, which provides a high-performance, binary protocol communication framework. It consists of:

1. **Protocol Buffers Definition**: Defines the service and message types
2. **gRPC Server**: Handles incoming requests and performs inference
3. **gRPC Client**: Sends requests to the server for inference

## Prerequisites

- Python 3.7+
- gRPC tools
- PyTorch

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Generate Python code from Protocol Buffer definitions:

```bash
python generate_proto.py
```

This will create `inference_pb2.py` and `inference_pb2_grpc.py` files.

## Usage

### Starting the Server

```bash
python server.py [--port PORT]
```

The server will start listening for gRPC requests on the specified port (default: 50051).

### Running the Client

```bash
# Single request mode
python client.py --server localhost:50051 --image test_image.jpg --mode single

# Benchmark mode
python client.py --server localhost:50051 --image test_image.jpg --mode benchmark --requests 100 --concurrency 10
```

## gRPC vs REST

gRPC offers several advantages over REST for ML model serving:

- **Performance**: Binary protocol with lower overhead and latency
- **Streaming**: Support for bidirectional streaming
- **Strong Typing**: Contract-based API with clear interface definition
- **Code Generation**: Automatic client/server code generation
- **HTTP/2**: Multiplexing, header compression, and binary framing

## Benchmarking

The client includes a benchmark mode that measures:

- Requests per second (RPS)
- End-to-end latency
- Server processing time
- Network latency

This allows for precise performance analysis and optimization.
