# Model Ensemble System

This module implements a model ensemble system that combines predictions from multiple machine learning models to improve accuracy and robustness.

## Features

- Combines predictions from multiple models
- Supports various aggregation methods:
  - Weighted average
  - Max voting
  - Softmax average
- Configurable model weights
- REST API for model serving

## Architecture

The system consists of three main components:

1. **ModelEnsemble**: Core component that handles prediction aggregation
2. **Server**: Flask-based web server that exposes the ensemble via a REST API
3. **Client**: Test client that demonstrates ensemble capabilities

## How It Works

Model ensembles work by combining the predictions of multiple models to produce a more accurate or robust final prediction. The ensemble leverages the strengths of different model architectures to improve overall performance.

In this implementation:

1. Multiple pre-trained models (ResNet50, DenseNet121, EfficientNetB0) are loaded
2. Each model processes the input independently
3. Predictions are aggregated using the specified method
4. The final ensemble prediction is returned

## Aggregation Methods

- **Weighted Average**: Each model's prediction is multiplied by its weight, then summed
- **Max Vote**: Each model votes for a class, and the class with the most votes wins
- **Softmax Average**: Softmax probabilities from each model are averaged

## Usage

### Starting the Server

```bash
python server.py
```

### Running the Client

```bash
# Detailed analysis of a single image
python client.py --url http://localhost:5000 --image test_image.jpg --mode detail

# Performance benchmarking
python client.py --url http://localhost:5000 --image test_image.jpg --mode benchmark --requests 50 --concurrency 10
```

## Benefits of Model Ensembles

- **Improved Accuracy**: Ensemble models typically outperform individual models
- **Reduced Overfitting**: Combining models can reduce the risk of overfitting
- **Better Generalization**: Ensembles often generalize better to new data
- **Increased Robustness**: Less susceptible to weaknesses of any single model

## Configuration

The ensemble behavior can be customized by adjusting:

- Model weights
- Aggregation method
- Model types included in the ensemble
