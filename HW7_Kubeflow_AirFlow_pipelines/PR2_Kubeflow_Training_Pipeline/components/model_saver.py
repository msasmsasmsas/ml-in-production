#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import NamedTuple
import os
import json
import shutil
from datetime import datetime
from google.cloud import storage

def save_model(
    model_path: str, 
    metrics_path: str, 
    output_path: str,
    model_name: str
) -> NamedTuple('Outputs', [('model_output_path', str), ('model_version', str)]):
    """
    Save the trained model and its metrics to the specified output path
    
    Args:
        model_path: Path to the trained model
        metrics_path: Path to the evaluation metrics
        output_path: Path to save the model artifacts
        model_name: Name of the model architecture
        
    Returns:
        model_output_path: Path where the model was saved
        model_version: Version assigned to the saved model
    """
    print(f"Saving model from {model_path} to {output_path}")
    
    # Create a timestamp-based version
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_version = f"{model_name}_{timestamp}"
    
    # Create a local directory to organize the files
    local_output_dir = f"/tmp/model_artifacts/{model_version}"
    os.makedirs(local_output_dir, exist_ok=True)
    
    # Copy the model file
    model_filename = os.path.basename(model_path)
    model_dest = os.path.join(local_output_dir, model_filename)
    shutil.copyfile(model_path, model_dest)
    
    # Load and copy the metrics
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    metrics_dest = os.path.join(local_output_dir, 'metrics.json')
    with open(metrics_dest, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # Load training config
    config_path = os.path.join(os.path.dirname(model_path), 'training_config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Create a combined model info file
    model_info = {
        'model_name': model_name,
        'model_version': model_version,
        'timestamp': timestamp,
        'metrics': metrics,
        'training_config': config
    }
    
    model_info_path = os.path.join(local_output_dir, 'model_info.json')
    with open(model_info_path, 'w') as f:
        json.dump(model_info, f, indent=4)
    
    # Copy any other relevant files
    history_path = os.path.join(os.path.dirname(model_path), 'training_history.csv')
    if os.path.exists(history_path):
        history_dest = os.path.join(local_output_dir, 'training_history.csv')
        shutil.copyfile(history_path, history_dest)
    
    # If output_path is a GCS path, upload the files
    if output_path.startswith('gs://'):
        # Parse the bucket and blob prefix
        bucket_name = output_path.split('gs://')[1].split('/')[0]
        prefix = '/'.join(output_path.split('gs://')[1].split('/')[1:])
        model_prefix = f"{prefix}/{model_version}" if prefix else model_version
        
        # Initialize GCS client
        storage_client = storage.Client()
        bucket = storage_client.get_bucket(bucket_name)
        
        # Upload all files in the local directory
        for root, dirs, files in os.walk(local_output_dir):
            for file in files:
                local_file = os.path.join(root, file)
                relative_path = os.path.relpath(local_file, local_output_dir)
                blob_name = f"{model_prefix}/{relative_path}"
                
                blob = bucket.blob(blob_name)
                blob.upload_from_filename(local_file)
                print(f"Uploaded {local_file} to gs://{bucket_name}/{blob_name}")
        
        # Set the model output path to the GCS path
        model_output_path = f"gs://{bucket_name}/{model_prefix}"
    else:
        # If it's a local path, create the directory and copy the files
        model_dir = os.path.join(output_path, model_version)
        os.makedirs(model_dir, exist_ok=True)
        
        # Copy all files from the temporary directory
        for root, dirs, files in os.walk(local_output_dir):
            for file in files:
                src_file = os.path.join(root, file)
                relative_path = os.path.relpath(src_file, local_output_dir)
                dst_file = os.path.join(model_dir, relative_path)
                
                # Create destination directory if it doesn't exist
                os.makedirs(os.path.dirname(dst_file), exist_ok=True)
                
                # Copy the file
                shutil.copyfile(src_file, dst_file)
                print(f"Copied {src_file} to {dst_file}")
        
        # Set the model output path to the local path
        model_output_path = model_dir
    
    print(f"Model and artifacts successfully saved to {model_output_path}")
    
    # Return the model output path and version
    from collections import namedtuple
    outputs = namedtuple('Outputs', ['model_output_path', 'model_version'])
    return outputs(model_output_path=model_output_path, model_version=model_version)

if __name__ == '__main__':
    # For testing the component locally
    save_model(
        model_path='/tmp/model/resnet18_model.pt',
        metrics_path='/tmp/evaluation/metrics.json',
        output_path='./saved_models',
        model_name='resnet18'
    )
