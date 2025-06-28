<!-- новлена версія для PR -->
# Scaling Infrastructure and Model Optimization

This module provides tools and configurations for scaling machine learning infrastructure and optimizing models for improved performance in production environments.

## Features

- Horizontal Pod Autoscaling for dynamic scaling based on load
- Asynchronous inference with message queues for high throughput
- Model optimization techniques (pruning, distillation, quantization)
- Comprehensive benchmarking tools for performance measurement

## Project Structure

```
HW12_Scaling_infra_and_model/
в”‚
в”њв”Ђв”Ђ PR1_horizontal_pod_autoscaling/    # Kubernetes configuration for autoscaling
в”‚   в”њв”Ђв”Ђ k8s-deployment.yaml            # Base deployment configuration
в”‚   в”њв”Ђв”Ђ hpa.yaml                       # HPA for CPU/memory-based scaling
в”‚   в”њв”Ђв”Ђ custom-metrics-hpa.yaml        # HPA with custom metrics
в”‚   в”њв”Ђв”Ђ prometheus-adapter-config.yaml # Configuration for custom metrics
в”‚   в””в”Ђв”Ђ README.md                      # Documentation
в”‚
в”њв”Ђв”Ђ PR2_async_inference/               # Async inference with message queues
в”‚   в”њв”Ђв”Ђ kafka_queue_service.py         # Kafka queue implementation
в”‚   в”њв”Ђв”Ђ async_fastapi_server.py        # FastAPI server with async endpoints
в”‚   в”њв”Ђв”Ђ model_worker.py                # Worker for processing inference requests
в”‚   в”њв”Ђв”Ђ docker-compose.yaml            # Docker Compose configuration
в”‚   в””в”Ђв”Ђ README.md                      # Documentation
в”‚
в”њв”Ђв”Ђ PR3_model_optimization/            # Model optimization techniques
в”‚   в”њв”Ђв”Ђ model_distillation.py          # Knowledge distillation implementation
в”‚   в”њв”Ђв”Ђ model_pruning.py               # Weight pruning implementation
в”‚   в”њв”Ђв”Ђ model_quantization.py          # Model quantization implementation
в”‚   в””в”Ђв”Ђ README.md                      # Documentation
в”‚
в””в”Ђв”Ђ PR4_model_benchmarking/            # Benchmarking and profiling tools
    в”њв”Ђв”Ђ benchmark_models.py            # Comprehensive model benchmarking
    в”њв”Ђв”Ђ model_flamegraph.py            # Performance profiling and flame graphs
    в””в”Ђв”Ђ README.md                      # Documentation
```

## Quick Start

### 1. Horizontal Pod Autoscaling

Deploy a model server with autoscaling capabilities:

```bash
# Create model storage
kubectl apply -f PR1_horizontal_pod_autoscaling/model-pvc.yaml

# Deploy the model server
kubectl apply -f PR1_horizontal_pod_autoscaling/k8s-deployment.yaml

# Apply autoscaling
kubectl apply -f PR1_horizontal_pod_autoscaling/hpa.yaml
```

### 2. Asynchronous Inference

Run an asynchronous inference service with Kafka queuing:

```bash
# Start the async inference stack
cd PR2_async_inference
docker-compose up -d

# Submit an inference request
curl -X POST "http://localhost:8000/inference" \
  -H "Content-Type: application/json" \
  -d '{"model_name":"image_classification","data":"your_data_here"}'
```

### 3. Model Optimization

Optimize a model using various techniques:

```bash
# Distill a model
python PR3_model_optimization/model_distillation.py \
  --teacher models/resnet50.pt \
  --save-path models/distilled_model.pt \
  --epochs 10

# Prune a model
python PR3_model_optimization/model_pruning.py \
  --model models/resnet50.pt \
  --pruning-method global \
  --amount 0.3 \
  --fine-tune \
  --save-path models/pruned_model.pt

# Quantize a model
python PR3_model_optimization/model_quantization.py \
  --model models/resnet50.pt \
  --quantization-method static \
  --save-path models/quantized_model.pt
```

### 4. Model Benchmarking

Benchmark optimized models:

```bash
python PR4_model_benchmarking/benchmark_models.py \
  --models "original:models/resnet50.pt,quantized:models/quantized_model.pt,pruned:models/pruned_model.pt,distilled:models/distilled_model.pt" \
  --report-format markdown
```

## Performance Results

Typical improvements achieved with optimization techniques:

| Technique | Model Size | Inference Speed | Accuracy Change |
|-----------|------------|-----------------|----------------|
| Original  | 100 MB     | 1.0x            | Baseline       |
| Distillation | ~50% | 2-3x | -0.5% to -2% |
| Pruning (30%) | ~70% | 1.3-1.5x | -0.5% to -1% |
| Quantization (INT8) | ~25% | 2.5-4x | -0.1% to -0.5% |
| Combined | ~15% | 3-5x | -1% to -3% |

## Requirements

- Kubernetes cluster for horizontal pod autoscaling
- Docker and Docker Compose for async inference
- PyTorch 1.8+ for model optimization and benchmarking
- CUDA-capable GPU for faster training and inference (optional)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

