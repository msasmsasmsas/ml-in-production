# PR2: Hyperparameter Search with Weights & Biases

This directory contains code for conducting hyperparameter searches for a DistilBERT model using Weights & Biases (W&B) sweeps.

## Files
- `hyperparam_search.py`: Script for hyperparameter search with W&B.
- `train.py`: Modified training script to accept hyperparameter configurations.
- `requirements.txt`: Dependencies.

## Usage
1. Install dependencies: `pip install -r requirements.txt`.
2. Set up W&B: `wandb login`.
3. Run hyperparameter search: `python hyperparam_search.py`.

## W&B Project
Hyperparameter search results are logged to the W&B project: [ml-in-production-hw5](https://wandb.ai/your-username/ml-in-production-hw5).