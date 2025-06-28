<!-- Updated version for PR -->
# Dynamic Request Batching

This module implements a dynamic request batching system for machine learning models. Dynamic batching is a technique that combines multiple inference requests into a single batch to improve throughput and resource utilization.

## Features

- Dynamically creates batches based on incoming requests
- Configurable maximum batch size and wait time
- Asynchronous request handling
- Synchronous API for easy integration

## Architecture

The system consists of three main components:

1. **DynamicBatcher**: Core component that aggregates requests and processes them in batches
2. **Server**: Flask-based web server that exposes the model via a REST API
3. **Client**: Test client that simulates concurrent requests

## How It Works

1. The DynamicBatcher runs a background thread that collects incoming requests
2. Requests are collected until either the maximum batch size is reached or the maximum wait time expires
3. Collected requests are processed as a single batch by the model
4. Results are disaggregated and returned to individual clients

## Usage

### Starting the Server

```bash
python server.py
```

### Running the Client for Testing

```bash
python client.py --url http://localhost:5000 --image test_image.jpg --requests 100 --concurrency 10
```

## Performance Benefits

Dynamic batching improves inference performance by:

- Increasing GPU utilization
- Reducing per-request overhead
- Optimizing throughput for varying load conditions
- Minimizing latency while maximizing resource efficiency

## Configuration Parameters

- `max_batch_size`: Maximum number of requests to process in a single batch
- `max_wait_time`: Maximum time to wait (in seconds) for additional requests before processing the batch

These parameters can be tuned based on your specific model characteristics and load patterns.

