# HW5: Training & Experiments

This directory contains the implementation of the training pipeline and experiments for the "Organizing Access to a RAG Repository of Information on Threats to Agricultural Crops" project. The tasks are organized as pull requests (PRs) to meet the requirements of the Machine Learning in Production course.

## Structure
- `hw5-pr1-training-wandb`: Training a DistilBERT model with Weights & Biases (W&B) logging.
- `hw5-pr2-hyperparameter-search-wandb`: Hyperparameter search using W&B sweeps.
- `hw5-pr3-model-card`: Model card creation using TensorFlow Model Card Toolkit.
- `hw5-pr4-mosaicbert`: (Optional) Replication of MosaicBERT pretraining.
- `hw5-pr5-nni-hyperparam`: (Optional) Hyperparameter search using Microsoft NNI.
- `hw5-pr6-distributed-training`: (Optional) Distributed training with PyTorch, Accelerate, and Ray.
- `data`: Directory for datasets (e.g., `dataset.csv`).

## W&B Project
All experiments are logged to the W&B project: [ml-in-production-hw5](https://wandb.ai/your-username/ml-in-production-hw5).

## Setup
1. Clone the repository: `git clone https://github.com/msasmsasmsas/ml-in-production.git`.
2. Navigate to the HW5 directory: `cd HW5_Training_Experiments`.
3. Install dependencies for each PR: `pip install -r <pr-directory>/requirements.txt`.
4. Set up W&B: `wandb login`.

## Usage
Refer to the `README.md` in each PR directory for specific instructions.

## Google Doc
The experiment section (experiment management tool and model card) is documented in the project design document: [Google Doc](https://docs.google.com/document/d/14vZZAcJgAqMXq3JPDxV4dyRlhJq6SXe2btOhH0gg8ug/edit?tab=t.0).