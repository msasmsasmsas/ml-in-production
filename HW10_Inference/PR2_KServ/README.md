# KServe Model Deployment

This directory contains code for deploying a machine learning model using KServe (formerly KFServing).

## Overview

KServe is a Kubernetes-based platform for serverless inference serving of machine learning models. It provides a unified, standard interface for serving models from frameworks like TensorFlow, PyTorch, scikit-learn, and more.

## Structure

- `model/` - Model implementation and artifacts
- `k8s/` - Kubernetes manifests for KServe deployment
- `client/` - Client code for interacting with the KServe deployment
- `tests/` - Tests for the deployment
- `requirements.txt` - Required dependencies

## Setup

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Building the Model Server

```bash
docker build -t kserve-resnet50:latest -f Dockerfile .
```

## Deploying to Kubernetes

```bash
# Apply the InferenceService
kubectl apply -f k8s/inference-service.yaml

# Check the status
kubectl get inferenceservices
```

## Running the client

```bash
python client/client.py --image path/to/image.jpg
```

## Running tests

```bash
pytest tests/
```

## Features

- Custom model serving with PyTorch
- Model explanation with LIME
- Request/response logging and monitoring
- Automatic scaling based on traffic
- GPU support for inference
