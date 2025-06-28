# Model Monitoring with Arize AI

This module integrates Arize AI for monitoring the agricultural threat detection model. Arize provides comprehensive monitoring capabilities including data drift detection, model performance tracking, and feature importance analysis.

## Features

- Real-time model performance monitoring
- Data drift detection for input features
- Prediction drift analysis
- Embedding visualization for images
- Model performance segmentation by crop type and region

## Setup

1. Create an Arize AI account at [https://arize.com](https://arize.com)
2. Create a `.env` file with your API credentials:

```
ARIZE_API_KEY=your-api-key
ARIZE_SPACE_KEY=your-space-key
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the example script to send synthetic data to Arize:

```bash
python arize_monitoring.py
```

This will generate synthetic crop data, train a model, and send predictions to Arize for monitoring.

## Integration with Production System

To integrate with the production system:

1. Import the `monitor_with_arize` function
2. Call it after making predictions, passing features, predictions, and actuals when available
3. Set up scheduled jobs to analyze historical data

## Dashboard Access

Access your monitoring dashboards at:
https://app.arize.com/models/crop_threat_detection_model
