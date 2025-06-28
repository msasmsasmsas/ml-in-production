# Asynchronous Inference Service with Kafka

This module provides an asynchronous model inference service using Apache Kafka as a message queue.

## Features

- Asynchronous model inference with Kafka queuing system
- Horizontal scaling of model workers
- Support for multiple model types (image classification, text processing)
- RESTful API with FastAPI for submitting and checking inference requests
- Webhook support for asynchronous notifications
- Docker Compose deployment with monitoring tools

## Architecture

The solution uses a queue-based architecture to decouple the API server from the model inference:

1. **FastAPI Server**: Handles client requests and response delivery
2. **Kafka Message Queue**: Manages the inference request/response queue
3. **Model Workers**: Process inference requests and return results
4. **Monitoring**: Prometheus and Grafana for performance monitoring

## Components

### 1. Kafka Queue Service

The `kafka_queue_service.py` provides the queue management and message handling between the API and workers.

### 2. Asynchronous FastAPI Server

The `async_fastapi_server.py` exposes RESTful endpoints for:
- Submitting inference requests
- Checking request status
- Retrieving inference results

### 3. Model Worker

The `model_worker.py` implements the inference processing logic, supporting various model types.

### 4. Docker Compose Configuration

Provides a complete deployment environment with Kafka, Zookeeper, and monitoring tools.

## Setup Instructions

### Prerequisites

- Docker and Docker Compose
- Python 3.8+
- PyTorch or other ML frameworks for your models

### Quick Start

1. **Clone the repository**:

```bash
git clone <repository-url>
cd <repository-dir>
```

2. **Place your model files in the models directory**:

```bash
mkdir -p models
# Copy your model files to the models directory
```

3. **Start the services with Docker Compose**:

```bash
docker-compose up -d
```

4. **Submit an inference request**:

```bash
curl -X POST "http://localhost:8000/inference" \
  -H "Content-Type: application/json" \
  -d '{"model_name":"image_classification","data":"your_base64_encoded_image"}'
```

5. **Check the request status**:

```bash
curl "http://localhost:8000/status/{request_id}"
```

6. **Retrieve the inference result**:

```bash
curl "http://localhost:8000/result/{request_id}"
```

## Monitoring

Access the monitoring tools at:
- Kafka UI: http://localhost:8080
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (admin/admin)

## Performance Considerations

- Adjust the number of workers based on your available resources
- Monitor Kafka lag to ensure workers can keep up with the request rate
- Consider batching for higher throughput

## Scaling

To scale the system horizontally:

```bash
docker-compose up -d --scale worker=5
```

This will start 5 worker containers to process requests in parallel.

## Integration with Kubernetes

For production deployments, consider using Kubernetes with the provided configuration files:

```bash
kubectl apply -f kubernetes/
```

## Security Considerations

- Enable Kafka authentication and encryption for production deployments
- Implement API authentication for the FastAPI server
- Consider using private networks for service communication

## License

This project is licensed under the MIT License - see the LICENSE file for details.
