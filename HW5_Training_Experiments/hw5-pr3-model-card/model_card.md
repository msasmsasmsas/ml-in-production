# Model Card: DistilBERT for Agricultural Threat Classification

## Model Details
- **Name**: DistilBERT for Agricultural Threat Classification
- **Overview**: This model classifies texts describing agricultural threats into three categories: weeds, pests, diseases.
- **Owners**: Your Name (your.email@example.com)
- **Version**: v1.0 (2025-05-12)
- **References**: Hugging Face Transformers: https://huggingface.co/docs/transformers
- **Path**: ./trained_model

## Quantitative Analysis
- **Performance Metrics**:
  - Accuracy: 0.85 (test_set)

## Training Data
- **Description**: Synthetic dataset of Ukrainian texts describing agricultural threats.
- **Size**: 1000 examples

## Evaluation Data
- **Description**: 20% split of the training dataset.