# HW5: Training & Experiments
# Эксперименты с обучением моделей для классификации рисков сельскохозяйственных культур
# Agricultural Risks Classification

This project implements a machine learning pipeline for classifying agricultural risks including plant diseases, pests, and weeds. The dataset contains information about various agricultural risks with images and detailed descriptions.

## Project Structure
Этот проект представляет собой набор инструментов для обучения и экспериментов с моделями машинного обучения для классификации рисков сельскохозяйственных культур.

## Структура проекта

- **PR1**: Обучение модели с использованием W&B для логирования экспериментов
- **PR2**: Поиск оптимальных гиперпараметров с помощью W&B Sweeps
- **PR3**: Создание карточки модели (Model Card)
- **PR4**: Репликация руководства MosaicBERT (опционально)
- **PR5**: Поиск гиперпараметров с использованием NNI (опционально)
- **PR6**: Распределенное обучение с PyTorch, Accelerate и Ray (опционально)

## Датасет

Датасет содержит информацию о рисках сельскохозяйственных культур и находится в папке `crawler/downloads`. Он включает данные о болезнях, вредителях и сорняках для различных сельскохозяйственных культур.

## Требования

- Python 3.8+
- PyTorch
- scikit-learn
- pandas
- numpy
- wandb
- transformers (для PR4)
- nni (для PR5)
- accelerate, ray (для PR6)
# Agricultural Risk Classification Project

This project implements training and experiment management for agricultural risk classification models. The dataset contains images of agricultural risks (diseases, pests, and weeds) affecting different crops.

## Project Structure

- **PR1**: Training model with W&B logging
- **PR2**: Hyperparameter search with W&B Sweeps
- **PR3**: Model Card creation for trained models
- **PR4**: MosaicBERT replication for agricultural risk data
- **PR5**: Hyperparameter optimization with Microsoft NNI
- **PR6**: Distributed training with PyTorch, Accelerate, and Ray

## Dataset

The dataset is located in `crawler/downloads` and contains:
- CSV files with information about agricultural risks
- Images of plant diseases, pests, and weeds
- Metadata about affected crops

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
## Установка зависимостей

```bash
pip install -r requirements.txt
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