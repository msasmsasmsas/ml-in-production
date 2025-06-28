# Model Benchmarking Tools

This module provides comprehensive tools for benchmarking and profiling machine learning models after optimization.

## Features

- Detailed performance metrics (speed, memory, accuracy)
- Visualization of benchmarking results
- Flame graphs for performance bottleneck identification
- Support for comparing multiple model variants
- Export results in various formats (JSON, Markdown, HTML)

## Components

### 1. Model Benchmarking

The `benchmark_models.py` script provides a comprehensive benchmarking suite for comparing multiple models.

**Metrics measured:**
- Model size (MB)
- Parameter count
- Memory usage
- Accuracy on test data
- Inference time (single sample and batch)
- Throughput (samples per second)

**Example Usage:**
```bash
python benchmark_models.py \
  --models "original:models/resnet50.pt,quantized:models/quantized_model.pt,pruned:models/pruned_model.pt,distilled:models/distilled_model.pt" \
  --dataset ./data/test \
  --batch-size 32 \
  --num-runs 100 \
  --save-dir benchmark_results \
  --report-format markdown
```

### 2. Performance Profiling

The `model_flamegraph.py` script creates detailed performance profiles and flame graphs to identify bottlenecks.

**Profiling tools supported:**
- PyTorch Profiler with TensorBoard visualization
- pyinstrument for Python-focused profiling
- cProfile for standard Python profiling

**Example Usage:**
```bash
python model_flamegraph.py \
  --model models/optimized_model.pt \
  --dataset ./data/test \
  --batch-size 1 \
  --save-dir profile_results \
  --profiler torch \
  --use-cuda
```

## Understanding the Results

### Benchmark Reports

The benchmarking tool generates comprehensive reports with the following sections:

1. **Summary Table**: Comparison of all models across key metrics
2. **Visualizations**: Bar charts and scatter plots showing relationships between metrics
3. **Detailed Logs**: Raw performance data for further analysis

### Flame Graphs

Flame graphs help identify performance bottlenecks in your model:

1. **Width**: Represents the time spent in each function
2. **Color**: Different colors represent different types of operations
3. **Stack**: The call stack shows how functions call each other

To view flame graphs:

```bash
tensorboard --logdir=profile_results/model_torch_trace_1234567890
```

## Interpreting Performance Metrics

### Size vs Speed Trade-offs

- **Model Size**: Smaller models load faster and use less memory
- **Inference Time**: Lower is better for real-time applications
- **Accuracy vs Speed**: Look for models in the upper-left quadrant of the accuracy/speed plot

### Memory Considerations

- **Peak Memory**: Critical for deployment on memory-constrained devices
- **Batch Size Impact**: Larger batches are more memory intensive but often more efficient

## Best Practices

1. **Always benchmark on the target hardware** when possible
2. **Use representative test data** that matches your production environment
3. **Run multiple benchmarking iterations** to account for variability
4. **Profile before and after optimization** to validate improvements
5. **Consider latency vs throughput** based on your application needs

## Requirements

- PyTorch 1.8+
- torchvision
- matplotlib
- tabulate
- tqdm
- pyinstrument (optional)
- tensorboard (for viewing flame graphs)

## Advanced Usage

### Custom Benchmarking

You can extend the `ModelBenchmark` class to add custom metrics specific to your application:

```python
class CustomBenchmark(ModelBenchmark):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def measure_custom_metric(self, model, device):
        # Your custom measurement code here
        return result
```

### Automating Benchmarks

For CI/CD integration, you can automate benchmarking by combining with model optimization:

```bash
# Example workflow
python model_quantization.py --model models/original.pt --save-path models/quantized.pt
python model_pruning.py --model models/original.pt --save-path models/pruned.pt
python model_distillation.py --teacher models/original.pt --save-path models/distilled.pt
python benchmark_models.py --models "original:models/original.pt,quantized:models/quantized.pt,pruned:models/pruned.pt,distilled:models/distilled.pt"
```
