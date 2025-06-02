#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import NamedTuple
import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.models as models
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(model_path: str, test_data_path: str) -> NamedTuple('Outputs', [('metrics_path', str)]):
    """
    Evaluate a trained model on a test dataset
    
    Args:
        model_path: Path to the trained model
        test_data_path: Path to the test data
        
    Returns:
        metrics_path: Path to the evaluation metrics
    """
    print(f"Evaluating model at {model_path}")
    
    # Create output directory
    output_dir = '/tmp/evaluation'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the training configuration
    config_path = os.path.join(os.path.dirname(model_path), 'training_config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    model_name = config['model_name']
    num_classes = config['num_classes']
    
    # Load dataset information
    dataset_info_path = os.path.join(os.path.dirname(test_data_path), '..', 'dataset_info.json')
    with open(dataset_info_path, 'r') as f:
        dataset_info = json.load(f)
    
    # Get class names if available
    class_names = dataset_info.get('classes', [f'Class_{i}' for i in range(num_classes)])
    
    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize model based on the architecture used during training
    if model_name == 'resnet18':
        model = models.resnet18(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'resnet50':
        model = models.resnet50(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'mobilenet_v2':
        model = models.mobilenet_v2(pretrained=False)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        raise ValueError(f"Unsupported model architecture: {model_name}")
    
    # Load model weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    # Load test dataloader
    test_loader = torch.load(os.path.join(test_data_path, 'test_loader.pth'))
    
    # Define loss function
    criterion = nn.CrossEntropyLoss()
    
    # Evaluation
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    
    # Store all predictions and true labels
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Track statistics
            test_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
            
            # Store predictions and true labels
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate test loss and accuracy
    test_loss = test_loss / len(test_loader.dataset)
    test_acc = test_correct / test_total
    
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
    
    # Generate classification report
    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_path = os.path.join(output_dir, 'classification_report.csv')
    report_df.to_csv(report_path)
    
    # Generate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    cm_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(cm_path)
    
    # Store evaluation metrics
    metrics = {
        'test_loss': test_loss,
        'test_accuracy': test_acc,
        'precision': report['weighted avg']['precision'],
        'recall': report['weighted avg']['recall'],
        'f1_score': report['weighted avg']['f1-score'],
        'classification_report_path': report_path,
        'confusion_matrix_path': cm_path
    }
    
    # Save metrics to JSON
    metrics_path = os.path.join(output_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print(f"Evaluation metrics saved to {metrics_path}")
    
    # Return the metrics path
    from collections import namedtuple
    outputs = namedtuple('Outputs', ['metrics_path'])
    return outputs(metrics_path=metrics_path)

if __name__ == '__main__':
    # For testing the component locally
    evaluate_model(model_path='/tmp/model/resnet18_model.pt', test_data_path='/tmp/processed_data/test')
