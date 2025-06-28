<!-- новлена версія для PR -->
<!-- новлена версія для PR -->
# Drift Detection for ML Pipeline

This directory contains the code for detecting data drift in our ML pipeline (Dagster) for both input features and model outputs.

## Components

- `drift_detection/`: Core drift detection logic
- `dagster_integration/`: Integration with Dagster pipeline
- `metrics/`: Custom metrics for measuring drift
- `visualization/`: Tools for visualizing detected drift

## Drift Detection Methods

- Statistical tests for distribution changes
- Model-based drift detection
- Feature importance monitoring
- Prediction distribution analysis

## Setup Instructions

1. Add drift detection components to your Dagster pipeline
2. Configure thresholds and notification settings
3. Deploy the pipeline with drift detection enabled

## Architecture

Drift detection is implemented as a separate step in the Dagster pipeline, analyzing both input data and model predictions to detect changes that might affect model performance.


