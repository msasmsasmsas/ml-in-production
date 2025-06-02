#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import NamedTuple
import os
import json
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torchvision import transforms, datasets
from PIL import Image

def preprocess_data(data_path: str, batch_size: int) -> NamedTuple('Outputs', [
    ('train_data_path', str), 
    ('val_data_path', str), 
    ('test_data_path', str)
]):
    """
    Preprocess the data and split into train, validation, and test sets
    
    Args:
        data_path: Path to the raw data
        batch_size: Batch size for data loaders
        
    Returns:
        train_data_path: Path to the training data
        val_data_path: Path to the validation data
        test_data_path: Path to the test data
    """
    print(f"Preprocessing data from {data_path}")
    
    # Create output directories
    output_dir = '/tmp/processed_data'
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    test_dir = os.path.join(output_dir, 'test')
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # Define data transformations
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Check if the data_path is a directory with class folders
    if os.path.isdir(data_path) and len([d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]) > 0:
        # Load the dataset as an ImageFolder
        dataset = datasets.ImageFolder(data_path)
        
        # Get class names
        classes = dataset.classes
        
        # Split into train, validation, and test sets
        train_indices, temp_indices = train_test_split(
            list(range(len(dataset))), test_size=0.3, random_state=42, stratify=dataset.targets
        )
        
        val_indices, test_indices = train_test_split(
            temp_indices, test_size=0.5, random_state=42, 
            stratify=[dataset.targets[i] for i in temp_indices]
        )
        
        # Create dataloaders
        train_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(data_path, transform=train_transform),
            batch_size=batch_size, sampler=torch.utils.data.SubsetRandomSampler(train_indices)
        )
        
        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(data_path, transform=val_test_transform),
            batch_size=batch_size, sampler=torch.utils.data.SubsetRandomSampler(val_indices)
        )
        
        test_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(data_path, transform=val_test_transform),
            batch_size=batch_size, sampler=torch.utils.data.SubsetRandomSampler(test_indices)
        )
        
        # Save class mapping
        class_mapping = {i: cls for i, cls in enumerate(classes)}
        with open(os.path.join(output_dir, 'class_mapping.json'), 'w') as f:
            json.dump(class_mapping, f)
        
        # Save dataset information
        dataset_info = {
            'num_classes': len(classes),
            'classes': classes,
            'train_samples': len(train_indices),
            'val_samples': len(val_indices),
            'test_samples': len(test_indices),
            'batch_size': batch_size
        }
        
        with open(os.path.join(output_dir, 'dataset_info.json'), 'w') as f:
            json.dump(dataset_info, f)
        
        # Save the dataloaders for subsequent steps
        torch.save(train_loader, os.path.join(train_dir, 'train_loader.pth'))
        torch.save(val_loader, os.path.join(val_dir, 'val_loader.pth'))
        torch.save(test_loader, os.path.join(test_dir, 'test_loader.pth'))
        
    else:
        # If not a directory with class folders, assume it's a custom dataset format
        print("Custom dataset format detected. Using dummy data for demonstration.")
        
        # Create dummy data for demonstration
        X = np.random.randn(100, 3, 224, 224).astype(np.float32)
        y = np.random.randint(0, 10, size=(100,)).astype(np.int64)
        
        # Split into train, validation, and test sets
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
        
        # Create TensorDatasets
        train_dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(X_train), torch.from_numpy(y_train)
        )
        
        val_dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(X_val), torch.from_numpy(y_val)
        )
        
        test_dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(X_test), torch.from_numpy(y_test)
        )
        
        # Create DataLoaders
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False
        )
        
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False
        )
        
        # Save dataset information
        dataset_info = {
            'num_classes': len(np.unique(y)),
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'test_samples': len(X_test),
            'batch_size': batch_size
        }
        
        with open(os.path.join(output_dir, 'dataset_info.json'), 'w') as f:
            json.dump(dataset_info, f)
        
        # Save the dataloaders for subsequent steps
        torch.save(train_loader, os.path.join(train_dir, 'train_loader.pth'))
        torch.save(val_loader, os.path.join(val_dir, 'val_loader.pth'))
        torch.save(test_loader, os.path.join(test_dir, 'test_loader.pth'))
    
    print(f"Data preprocessing completed. Processed data saved to {output_dir}")
    
    # Return the output paths
    from collections import namedtuple
    outputs = namedtuple('Outputs', ['train_data_path', 'val_data_path', 'test_data_path'])
    return outputs(train_data_path=train_dir, val_data_path=val_dir, test_data_path=test_dir)

if __name__ == '__main__':
    # For testing the component locally
    preprocess_data(data_path='./sample_data', batch_size=32)
