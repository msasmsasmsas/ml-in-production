#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import NamedTuple
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.models as models
import time
import pandas as pd

def train_model(
    train_data_path: str, 
    val_data_path: str, 
    epochs: int, 
    learning_rate: float, 
    batch_size: int,
    model_name: str
) -> NamedTuple('Outputs', [('model_path', str), ('training_logs_path', str)]):
    """
    Train a neural network model on the provided dataset
    
    Args:
        train_data_path: Path to the training data
        val_data_path: Path to the validation data
        epochs: Number of training epochs
        learning_rate: Learning rate for the optimizer
        batch_size: Batch size for training
        model_name: Name of the model architecture to use
        
    Returns:
        model_path: Path to the trained model
        training_logs_path: Path to the training logs
    """
    print(f"Training model with {model_name} architecture for {epochs} epochs")
    
    # Create output directory
    output_dir = '/tmp/model'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the dataset information
    dataset_info_path = os.path.join(os.path.dirname(train_data_path), '..', 'dataset_info.json')
    with open(dataset_info_path, 'r') as f:
        dataset_info = json.load(f)
    
    num_classes = dataset_info['num_classes']
    
    # Load dataloaders
    train_loader = torch.load(os.path.join(train_data_path, 'train_loader.pth'))
    val_loader = torch.load(os.path.join(val_data_path, 'val_loader.pth'))
    
    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize model based on the specified architecture
    if model_name == 'resnet18':
        model = models.resnet18(pretrained=True)
        # Replace the final fully connected layer
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'resnet50':
        model = models.resnet50(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'mobilenet_v2':
        model = models.mobilenet_v2(pretrained=True)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        raise ValueError(f"Unsupported model architecture: {model_name}")
    
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=3, verbose=True
    )
    
    # Training and validation history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'learning_rate': [],
        'epoch_time': []
    }
    
    # Training loop
    for epoch in range(epochs):
        start_time = time.time()
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Track statistics
            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        epoch_train_loss = train_loss / len(train_loader.dataset)
        epoch_train_acc = train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                # Track statistics
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        epoch_val_loss = val_loss / len(val_loader.dataset)
        epoch_val_acc = val_correct / val_total
        
        # Update learning rate
        scheduler.step(epoch_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Record time taken
        epoch_time = time.time() - start_time
        
        # Update history
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)
        history['learning_rate'].append(current_lr)
        history['epoch_time'].append(epoch_time)
        
        print(f"Epoch {epoch+1}/{epochs} - "
              f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f}, "
              f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}, "
              f"LR: {current_lr:.6f}, Time: {epoch_time:.2f}s")
    
    # Save the trained model
    model_save_path = os.path.join(output_dir, f"{model_name}_model.pt")
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")
    
    # Save training history
    history_df = pd.DataFrame(history)
    history_path = os.path.join(output_dir, 'training_history.csv')
    history_df.to_csv(history_path, index=False)
    print(f"Training history saved to {history_path}")
    
    # Save training configuration
    config = {
        'model_name': model_name,
        'epochs': epochs,
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'num_classes': num_classes,
        'device': str(device),
        'final_train_loss': history['train_loss'][-1],
        'final_train_acc': history['train_acc'][-1],
        'final_val_loss': history['val_loss'][-1],
        'final_val_acc': history['val_acc'][-1]
    }
    
    config_path = os.path.join(output_dir, 'training_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    # Return the model path and training logs path
    from collections import namedtuple
    outputs = namedtuple('Outputs', ['model_path', 'training_logs_path'])
    return outputs(model_path=model_save_path, training_logs_path=history_path)

if __name__ == '__main__':
    # For testing the component locally
    train_model(
        train_data_path='/tmp/processed_data/train',
        val_data_path='/tmp/processed_data/val',
        epochs=5,
        learning_rate=0.001,
        batch_size=32,
        model_name='resnet18'
    )
