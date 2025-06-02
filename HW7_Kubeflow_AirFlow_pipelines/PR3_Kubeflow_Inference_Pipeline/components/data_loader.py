#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import NamedTuple
import os
import json
from google.cloud import storage
import torch
from torchvision import transforms, datasets
from PIL import Image
import pandas as pd

def load_inference_data(data_path: str, batch_size: int) -> NamedTuple('Outputs', [('data_path', str)]):
    """
    Load data for inference
    
    Args:
        data_path: Path to the data for inference, can be a GCS bucket or local path
        batch_size: Batch size for inference
        
    Returns:
        data_path: Path to the loaded data for inference
    """
    print(f"Loading inference data from {data_path}")
    
    # Create output directory if it doesn't exist
    output_dir = '/tmp/inference_data'
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if the path is a GCS path
    if data_path.startswith('gs://'):
        # Parse the bucket and blob names
        bucket_name = data_path.split('gs://')[1].split('/')[0]
        prefix = '/'.join(data_path.split('gs://')[1].split('/')[1:])
        
        # Initialize GCS client
        storage_client = storage.Client()
        bucket = storage_client.get_bucket(bucket_name)
        
        # List all blobs in the specified path
        blobs = bucket.list_blobs(prefix=prefix)
        
        # Download all files
        for blob in blobs:
            if not blob.name.endswith('/'):  # Skip directories
                destination_file = os.path.join(output_dir, os.path.basename(blob.name))
                blob.download_to_filename(destination_file)
                print(f"Downloaded {blob.name} to {destination_file}")
                
        data_path = output_dir
    else:
        # If local path, verify it exists
        if not os.path.exists(data_path):
            raise ValueError(f"Data path {data_path} does not exist")
    
    # Create standard preprocessing transformation for inference
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Process the data based on its format
    if os.path.isdir(data_path):
        # Check if there's a CSV file with image paths
        csv_files = [f for f in os.listdir(data_path) if f.endswith('.csv')]
        if csv_files:
            csv_path = os.path.join(data_path, csv_files[0])
            df = pd.read_csv(csv_path)
            
            # Check if the CSV has an image path column
            img_path_cols = [col for col in df.columns if 'image' in col.lower() or 'path' in col.lower()]
            if img_path_cols:
                img_path_col = img_path_cols[0]
                print(f"Using column '{img_path_col}' as image path")
                
                # Create a dataset from the image paths
                img_paths = df[img_path_col].tolist()
                
                # Verify image paths (first 5 for speed)
                for i, img_path in enumerate(img_paths[:5]):
                    if not os.path.isabs(img_path):
                        img_path = os.path.join(data_path, img_path)
                    
                    if not os.path.exists(img_path):
                        print(f"Warning: Image {img_path} does not exist")
                
                # Create a simple dataset metadata
                dataset_info = {
                    'type': 'csv_image_paths',
                    'num_samples': len(img_paths),
                    'csv_path': csv_path,
                    'img_path_column': img_path_col
                }
            else:
                print("CSV file doesn't contain recognizable image path column")
                dataset_info = {
                    'type': 'unknown',
                    'csv_path': csv_path
                }
        else:
            # Check if it's a directory with images
            image_files = [f for f in os.listdir(data_path) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
            
            if image_files:
                print(f"Found {len(image_files)} images in the directory")
                
                # Create a dataset from the images
                img_paths = [os.path.join(data_path, f) for f in image_files]
                
                # Create a dataloader for the images
                try:
                    dataset = datasets.ImageFolder(data_path, transform=preprocess)
                    dataloader = torch.utils.data.DataLoader(
                        dataset, batch_size=batch_size, shuffle=False
                    )
                    
                    # Save the dataloader
                    torch.save(dataloader, os.path.join(output_dir, 'inference_dataloader.pth'))
                    
                    # Create dataset metadata
                    dataset_info = {
                        'type': 'image_folder',
                        'num_samples': len(dataset),
                        'classes': dataset.classes,
                        'batch_size': batch_size
                    }
                except Exception as e:
                    print(f"Failed to create ImageFolder dataset: {e}")
                    print("Creating a simple list of image paths instead")
                    
                    # Create a simple dataset metadata
                    dataset_info = {
                        'type': 'image_paths',
                        'num_samples': len(img_paths),
                        'image_paths': img_paths
                    }
            else:
                print("No image files found in the directory")
                dataset_info = {
                    'type': 'unknown',
                    'directory': data_path
                }
    else:
        # Single file, check if it's an image or a text file with paths
        if data_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            print("Single image file for inference")
            img_paths = [data_path]
            
            # Create a simple dataset metadata
            dataset_info = {
                'type': 'single_image',
                'image_path': data_path
            }
        elif data_path.lower().endswith(('.csv', '.txt')):
            print("Text file with image paths")
            
            if data_path.lower().endswith('.csv'):
                df = pd.read_csv(data_path)
                
                # Check if the CSV has an image path column
                img_path_cols = [col for col in df.columns if 'image' in col.lower() or 'path' in col.lower()]
                if img_path_cols:
                    img_path_col = img_path_cols[0]
                    print(f"Using column '{img_path_col}' as image path")
                    
                    # Create a dataset from the image paths
                    img_paths = df[img_path_col].tolist()
                    
                    # Create a simple dataset metadata
                    dataset_info = {
                        'type': 'csv_image_paths',
                        'num_samples': len(img_paths),
                        'csv_path': data_path,
                        'img_path_column': img_path_col
                    }
                else:
                    print("CSV file doesn't contain recognizable image path column")
                    dataset_info = {
                        'type': 'unknown',
                        'csv_path': data_path
                    }
            else:
                # Assume it's a text file with one image path per line
                with open(data_path, 'r') as f:
                    img_paths = [line.strip() for line in f if line.strip()]
                
                # Create a simple dataset metadata
                dataset_info = {
                    'type': 'txt_image_paths',
                    'num_samples': len(img_paths),
                    'txt_path': data_path
                }
        else:
            print(f"Unsupported file type: {data_path}")
            dataset_info = {
                'type': 'unknown',
                'file_path': data_path
            }
    
    # Save dataset info
    dataset_info_path = os.path.join(output_dir, 'dataset_info.json')
    with open(dataset_info_path, 'w') as f:
        json.dump(dataset_info, f, indent=4)
    
    print(f"Inference data loaded. Info saved to {dataset_info_path}")
    
    # Return the output path
    from collections import namedtuple
    outputs = namedtuple('Outputs', ['data_path'])
    return outputs(data_path=output_dir)

if __name__ == '__main__':
    # For testing the component locally
    load_inference_data(data_path='./sample_inference_data', batch_size=32)
