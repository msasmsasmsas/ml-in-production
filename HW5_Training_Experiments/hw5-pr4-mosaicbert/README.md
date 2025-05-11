# PR4: MosaicBERT Replication

This directory contains code to replicate the MosaicBERT pretraining process as described in the Databricks blog.

## Files
- `mosaicbert_train.py`: Script for pretraining a BERT model from scratch.
- `requirements.txt`: Dependencies.

## Usage
1. Install dependencies: `pip install -r requirements.txt`.
2. Set up W&B: `wandb login`.
3. Run training: `python mosaicbert_train.py`.

## W&B Project
Experiments are logged to the W&B project: [ml-in-production-hw5](https://wandb.ai/your-username/ml-in-production-hw5).