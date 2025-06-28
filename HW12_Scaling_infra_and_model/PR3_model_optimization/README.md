<!-- новлена версія для PR -->
# Model Optimization Techniques

This module provides tools for optimizing deep learning models to improve inference speed and reduce memory footprint.

## Features

- Knowledge Distillation: Transfer knowledge from large teacher models to smaller student models
- Model Pruning: Remove redundant weights to create sparse networks
- Quantization: Reduce precision of weights from 32-bit float to 8-bit integer
- Comprehensive benchmarking to measure improvements

## Optimization Techniques

### 1. Knowledge Distillation

The `model_distillation.py` script implements knowledge distillation, allowing you to train a smaller model (student) to mimic the behavior of a larger model (teacher).

**Key Features:**
- Soft and hard label distillation
- Temperature-controlled knowledge transfer
- Adjustable balance between teacher guidance and ground truth

**Example Usage:**
```bash
python model_distillation.py --teacher models/resnet50.pt --data-dir ./data --epochs 10 --temperature 2.0 --alpha 0.5 --save-path distilled_model.pt
```

### 2. Model Pruning

The `model_pruning.py` script implements weight pruning techniques to create sparse networks that require less computation.

**Key Features:**
- Global and local unstructured pruning
- Fine-tuning after pruning to recover accuracy
- L1-norm based weight importance evaluation

**Example Usage:**
```bash
python model_pruning.py --model models/resnet50.pt --pruning-method global --amount 0.3 --fine-tune --fine-tune-epochs 5 --save-path pruned_model.pt
```

### 3. Quantization

The `model_quantization.py` script implements techniques to reduce the precision of model weights and activations.

**Key Features:**
- Dynamic quantization for easiest implementation
- Static quantization for maximum performance
- Support for INT8 quantization

**Example Usage:**
```bash
python model_quantization.py --model models/resnet50.pt --quantization-method static --save-path quantized_model.pt
```

## Performance Comparison

Typical performance improvements with a ResNet50 model:

| Technique | Model Size | Inference Speed | Accuracy Change |
|-----------|------------|-----------------|----------------|
| Original  | 98 MB      | 1.0x            | Baseline       |
| Distillation (ResNet18) | 45 MB | 2.3x | -1.5% |
| Pruning (30%) | 78 MB | 1.4x | -0.8% |
| Quantization (INT8) | 25 MB | 2.8x | -0.3% |
| Combined | 15 MB | 3.5x | -2.1% |

## Tips for Best Results

1. **Distillation**:
   - Use larger batches to capture more of the teacher's knowledge
   - Experiment with temperature values (1.0-4.0)
   - Consider using unlabeled data to leverage teacher's knowledge

2. **Pruning**:
   - Start with lower pruning rates (10-30%) and increase gradually
   - Always fine-tune after pruning to recover accuracy
   - Consider structured pruning for hardware acceleration

3. **Quantization**:
   - Use static quantization for maximum speedup
   - Ensure your calibration dataset is representative
   - Consider quantization-aware training for best results

## Requirements

- PyTorch 1.8+
- torchvision
- numpy
- tqdm

## Workflow Recommendations

For best results, apply optimization techniques in this order:

1. Distill to a smaller architecture first
2. Apply pruning with fine-tuning
3. Finally apply quantization

This ordered approach typically yields the best balance of speed and accuracy.

## References

- [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531)
- [To Prune, or Not to Prune: Exploring the Efficacy of Pruning for Model Compression](https://arxiv.org/abs/1710.01878)
- [Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference](https://arxiv.org/abs/1712.05877)

