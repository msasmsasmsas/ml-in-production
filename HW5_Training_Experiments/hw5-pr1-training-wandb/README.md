# PR1: Training with Weights & Biases

This directory contains code for training a DistilBERT model for text classification (categorizing agricultural threats: weeds, pests, diseases) using Weights & Biases (W&B) for experiment logging.

## Files
- `train.py`: Script for training the model with W&B logging.
- `dataset.py`: Data loading and preprocessing.
- `model.py`: Model definition.
- `requirements.txt`: Dependencies.

## Usage
1. Install dependencies: `pip install -r requirements.txt`.
2. Set up W&B: `wandb login`.
3. Run training: `python train.py`.

## W&B Project
Experiments are logged to the W&B project: [ml-in-production-hw5](https://wandb.ai/your-username/ml-in-production-hw5).