#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import NamedTuple
import os
import json
import torch
import torch.nn as nn
import torchvision.models as models
from google.cloud import storage

def load_model(model_path: str, model_info_path: str) -> NamedTuple('Outputs', [
    ('prepared_model_path', str), 
    ('model_config_path', str)
]):
    """
    Load a trained model for inference
    
    Args:
        model_path: Path to the trained model
        model_info_path: Path to the model info JSON file
        
    Returns:
        prepared_model_path: Path to the prepared model
        model_config_path: Path to the model configuration
    """
    print(f"Loading model from {model_path}")
    
    # Create output directory if it doesn't exist
    output_dir = '/tmp/inference_model'
    os.makedirs(output_dir, exist_ok=True)
    
    # Download model if it's on GCS
    if model_path.startswith('gs://'):
        # Parse the bucket and blob name
        bucket_name = model_path.split('gs://')[1].split('/')[0]
        blob_name = '/'.join(model_path.split('gs://')[1].split('/')[1:])
        
        # Initialize GCS client
        storage_client = storage.Client()
        bucket = storage_client.get_bucket(bucket_name)
        blob = bucket.blob(blob_name)
        
        # Download model file
        local_model_path = os.path.join(output_dir, os.path.basename(model_path))
        blob.download_to_filename(local_model_path)
        print(f"Downloaded model from {model_path} to {local_model_path}")
        
        model_path = local_model_path
    else:
        # If local path, verify it exists and copy
        if not os.path.exists(model_path):
            raise ValueError(f"Model path {model_path} does not exist")
        
        local_model_path = os.path.join(output_dir, os.path.basename(model_path))
        torch.save(torch.load(model_path, map_location='cpu'), local_model_path)
        print(f"Copied model from {model_path} to {local_model_path}")
        
        model_path = local_model_path
    
    # Download model info if it's on GCS
    if model_info_path.startswith('gs://'):
        # Parse the bucket and blob name
        bucket_name = model_info_path.split('gs://')[1].split('/')[0]
        blob_name = '/'.join(model_info_path.split('gs://')[1].split('/')[1:])
        
        # Initialize GCS client if not already done
        if not 'storage_client' in locals():
            storage_client = storage.Client()
            bucket = storage_client.get_bucket(bucket_name)
        else:
            bucket = storage_client.get_bucket(bucket_name)
            
        blob = bucket.blob(blob_name)
        
        # Download model info file
        local_info_path = os.path.join(output_dir, 'model_info.json')
        blob.download_to_filename(local_info_path)
        print(f"Downloaded model info from {model_info_path} to {local_info_path}")
        
        with open(local_info_path, 'r') as f:
            model_info = json.load(f)
    else:
        # If local path, verify it exists
        if not os.path.exists(model_info_path):
            # If model info path doesn't exist, try to infer model type from the model file
            print(f"Model info path {model_info_path} does not exist. Trying to infer model type.")
            
            # Create a basic model info
            model_info = {
                'model_name': os.path.basename(model_path).split('_')[0],  # Guess from filename
                'model_version': 'unknown',
                'timestamp': 'unknown',
                'training_config': {
                    'num_classes': 1000,  # Assume ImageNet classes if not specified
                    'model_name': os.path.basename(model_path).split('_')[0]
                }
            }
        else:
            # Load model info
            with open(model_info_path, 'r') as f:
                model_info = json.load(f)
    
    # Extract model configuration
    model_name = model_info.get('model_name', model_info.get('training_config', {}).get('model_name', 'resnet18'))
    num_classes = model_info.get('training_config', {}).get('num_classes', 1000)
    
    # Initialize the model architecture
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
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    # Save prepared model
    prepared_model_path = os.path.join(output_dir, 'prepared_model.pt')
    torch.save(model.state_dict(), prepared_model_path)
    
    # Save model configuration
    config = {
        'model_name': model_name,
        'num_classes': num_classes,
        'model_version': model_info.get('model_version', 'unknown'),
        'input_shape': [3, 224, 224],  # Standard ImageNet input size
        'device': str(device),
        'class_names': model_info.get('class_names', [f'class_{i}' for i in range(num_classes)])
    }
    
    config_path = os.path.join(output_dir, 'model_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    print(f"Model prepared for inference. Config saved to {config_path}")
    
    # Return paths
    from collections import namedtuple
    outputs = namedtuple('Outputs', ['prepared_model_path', 'model_config_path'])
    return outputs(prepared_model_path=prepared_model_path, model_config_path=config_path)

if __name__ == '__main__':
    # For testing the component locally
    load_model(
        model_path='./sample_models/resnet18_model.pt',
        model_info_path='./sample_models/model_info.json'
    )
