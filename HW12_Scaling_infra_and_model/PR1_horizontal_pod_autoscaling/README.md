<!-- новлена версія для PR -->
# Horizontal Pod Autoscaling for Model Server

This module provides configuration for automatically scaling model server deployments based on load metrics.

## Features

- Standard CPU and memory-based autoscaling
- Custom metrics autoscaling (request rate, latency)
- Prometheus integration for metrics collection
- Configurable scaling policies

## Components

### 1. Kubernetes Deployment

The `k8s-deployment.yaml` file contains the base deployment configuration for the model server with:
- Resource requests and limits
- Health checks
- Volume mounts for model storage

### 2. Horizontal Pod Autoscaler (HPA)

Two HPA configurations are provided:

- `hpa.yaml`: Basic CPU and memory-based autoscaling
- `custom-metrics-hpa.yaml`: Advanced autoscaling based on custom metrics like:
  - Inference requests per second
  - P95 inference latency
  - Total service request rate

### 3. Prometheus Adapter

The `prometheus-adapter-config.yaml` configures the Prometheus Adapter to expose custom metrics to the Kubernetes metrics API.

## Setup Instructions

### Prerequisites

- Kubernetes cluster with metrics-server installed
- Prometheus and Prometheus Adapter for custom metrics
- Docker image with your model server

### Deployment Steps

1. **Create the model storage PVC:**

```bash
kubectl apply -f model-pvc.yaml
```

2. **Deploy the model server:**

```bash
kubectl apply -f k8s-deployment.yaml
```

3. **Apply the HPA configuration:**

For basic resource-based autoscaling:
```bash
kubectl apply -f hpa.yaml
```

For custom metrics-based autoscaling (requires Prometheus Adapter):
```bash
kubectl apply -f prometheus-adapter-config.yaml
kubectl apply -f custom-metrics-hpa.yaml
```

### Monitoring

Check the status of your HPA:
```bash
kubectl get hpa model-server-hpa -w
```

View scaling events:
```bash
kubectl describe hpa model-server-hpa
```

## Testing Autoscaling

To test the autoscaling functionality, you can use a load testing tool like `hey` or `locust`:

```bash
hey -z 5m -c 50 -q 10 http://<service-ip>/predict
```

You should see the number of pods increasing as the load increases, and decreasing when the load subsides.

## Best Practices

1. Set appropriate resource requests based on actual model needs
2. Configure stabilization windows to prevent thrashing
3. Use custom metrics for more targeted scaling
4. Monitor scaling events to fine-tune HPA parameters
5. Consider using Pod Disruption Budgets (PDBs) for higher availability

