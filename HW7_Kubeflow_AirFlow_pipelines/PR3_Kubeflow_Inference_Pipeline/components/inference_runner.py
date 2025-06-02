#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import NamedTuple
import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.models as models
import time
import pandas as pd
import numpy as np
from PIL import Image
from torchvision import transforms

def run_inference(
    data_path: str, 
    model_path: str,
    model_config_path: str,
    batch_size: int,
    confidence_threshold: float = 0.5
) -> NamedTuple('Outputs', [('predictions_path', str), ('performance_metrics_path', str)]):
    """
    Run inference on the provided data using the loaded model
    
    Args:
        data_path: Path to the data for inference
        model_path: Path to the prepared model
        model_config_path: Path to the model configuration
        batch_size: Batch size for inference
        confidence_threshold: Confidence threshold for predictions
        
    Returns:
        predictions_path: Path to the inference predictions
        performance_metrics_path: Path to the inference performance metrics
    """
    print(f"Running inference on data at {data_path} with model at {model_path}")
    
    # Create output directory if it doesn't exist
    output_dir = '/tmp/inference_results'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model configuration
    with open(model_config_path, 'r') as f:
        model_config = json.load(f)
    
    model_name = model_config['model_name']
    num_classes = model_config['num_classes']
    class_names = model_config.get('class_names', [f'class_{i}' for i in range(num_classes)])
    device = torch.device(model_config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    
    # Load dataset information
    dataset_info_path = os.path.join(data_path, 'dataset_info.json')
    with open(dataset_info_path, 'r') as f:
        dataset_info = json.load(f)
    
    dataset_type = dataset_info['type']
    
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
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    # Define preprocessing transformation
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Initialize performance metrics
    total_time = 0
    num_samples = 0
    all_predictions = []
    
    # Run inference based on dataset type
    if dataset_type == 'image_folder':
        # Check if there's a saved dataloader
        dataloader_path = os.path.join(data_path, 'inference_dataloader.pth')
        if os.path.exists(dataloader_path):
            # Load the saved dataloader
            dataloader = torch.load(dataloader_path)
            
            # Run inference on the dataloader
            with torch.no_grad():
                for i, (inputs, _) in enumerate(dataloader):
                    start_time = time.time()
                    
                    inputs = inputs.to(device)
                    outputs = model(inputs)
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)
                    
                    batch_time = time.time() - start_time
                    total_time += batch_time
                    num_samples += inputs.size(0)
                    
                    # Process batch predictions
                    for j in range(inputs.size(0)):
                        probs, indices = torch.topk(probabilities[j], 5)
                        
                        # Convert to numpy for easier handling
                        probs = probs.cpu().numpy()
                        indices = indices.cpu().numpy()
                        
                        # Get the top classes and their probabilities
                        top_classes = [class_names[idx] for idx in indices]
                        top_probs = probs.tolist()
                        
                        # Create prediction record
                        prediction = {
                            'batch_idx': i,
                            'sample_idx': j,
                            'top_classes': top_classes,
                            'top_probabilities': top_probs,
                            'predicted_class': top_classes[0],
                            'confidence': top_probs[0]
                        }
                        
                        all_predictions.append(prediction)
                    
                    # Print progress
                    if (i + 1) % 10 == 0 or (i + 1) == len(dataloader):
                        print(f"Processed batch {i+1}/{len(dataloader)}")
        else:
            print("No dataloader found, processing individual images")
            # Process as individual images
            image_files = [f for f in os.listdir(data_path) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
            
            for i, img_file in enumerate(image_files):
                img_path = os.path.join(data_path, img_file)
                
                try:
                    # Load and preprocess the image
                    with Image.open(img_path) as img:
                        img_tensor = preprocess(img).unsqueeze(0).to(device)
                    
                    # Run inference
                    start_time = time.time()
                    with torch.no_grad():
                        outputs = model(img_tensor)
                        probabilities = torch.nn.functional.softmax(outputs, dim=1)
                    
                    batch_time = time.time() - start_time
                    total_time += batch_time
                    num_samples += 1
                    
                    # Process prediction
                    probs, indices = torch.topk(probabilities[0], 5)
                    
                    # Convert to numpy for easier handling
                    probs = probs.cpu().numpy()
                    indices = indices.cpu().numpy()
                    
                    # Get the top classes and their probabilities
                    top_classes = [class_names[idx] for idx in indices]
                    top_probs = probs.tolist()
                    
                    # Create prediction record
                    prediction = {
                        'image_path': img_path,
                        'top_classes': top_classes,
                        'top_probabilities': top_probs,
                        'predicted_class': top_classes[0],
                        'confidence': top_probs[0]
                    }
                    
                    all_predictions.append(prediction)
                    
                    # Print progress
                    if (i + 1) % 100 == 0 or (i + 1) == len(image_files):
                        print(f"Processed image {i+1}/{len(image_files)}")
                except Exception as e:
                    print(f"Error processing image {img_path}: {e}")
    else:
        # For other dataset types, process as needed
        print(f"Processing dataset of type: {dataset_type}")
        
        # Example handling for csv_image_paths
        if dataset_type == 'csv_image_paths':
            csv_path = dataset_info.get('csv_path')
            img_path_col = dataset_info.get('img_path_column')
            
            if csv_path and img_path_col:
                # Load the CSV
                if os.path.exists(csv_path):
                    df = pd.read_csv(csv_path)
                else:
                    # Try finding the CSV in the data path
                    csv_files = [f for f in os.listdir(data_path) if f.endswith('.csv')]
                    if csv_files:
                        csv_path = os.path.join(data_path, csv_files[0])
                        df = pd.read_csv(csv_path)
                    else:
                        raise ValueError(f"CSV file not found at {csv_path}")
                
                # Process each image
                for i, row in df.iterrows():
                    img_path = row[img_path_col]
                    
                    # Check if path is relative and make it absolute
                    if not os.path.isabs(img_path):
                        img_path = os.path.join(os.path.dirname(csv_path), img_path)
                    
                    try:
                        # Load and preprocess the image
                        with Image.open(img_path) as img:
                            img_tensor = preprocess(img).unsqueeze(0).to(device)
                        
                        # Run inference
                        start_time = time.time()
                        with torch.no_grad():
                            outputs = model(img_tensor)
                            probabilities = torch.nn.functional.softmax(outputs, dim=1)
                        
                        batch_time = time.time() - start_time
                        total_time += batch_time
                        num_samples += 1
                        
                        # Process prediction
                        probs, indices = torch.topk(probabilities[0], 5)
                        
                        # Convert to numpy for easier handling
                        probs = probs.cpu().numpy()
                        indices = indices.cpu().numpy()
                        
                        # Get the top classes and their probabilities
                        top_classes = [class_names[idx] for idx in indices]
                        top_probs = probs.tolist()
                        
                        # Create prediction record
                        prediction = {
                            'image_path': img_path,
                            'top_classes': top_classes,
                            'top_probabilities': top_probs,
                            'predicted_class': top_classes[0],
                            'confidence': top_probs[0]
                        }
                        
                        # Add other columns from the CSV
                        for col in df.columns:
                            if col != img_path_col:
                                prediction[col] = row[col]
                        
                        all_predictions.append(prediction)
                        
                        # Print progress
                        if (i + 1) % 100 == 0 or (i + 1) == len(df):
                            print(f"Processed image {i+1}/{len(df)}")
                    except Exception as e:
                        print(f"Error processing image {img_path}: {e}")
        else:
            print(f"Unsupported dataset type: {dataset_type}")
    
    # Calculate performance metrics
    if num_samples > 0:
        avg_time_per_sample = total_time / num_samples
        throughput = num_samples / total_time if total_time > 0 else 0
    else:
        avg_time_per_sample = 0
        throughput = 0
    
    # Filter predictions by confidence threshold
    confident_predictions = [p for p in all_predictions if p['confidence'] >= confidence_threshold]
    
    # Save performance metrics
    performance_metrics = {
        'total_samples': num_samples,
        'total_inference_time': total_time,
        'average_time_per_sample': avg_time_per_sample,
        'throughput_samples_per_second': throughput,
        'confident_predictions': len(confident_predictions),
        'confidence_threshold': confidence_threshold
    }
    
    performance_metrics_path = os.path.join(output_dir, 'performance_metrics.json')
    with open(performance_metrics_path, 'w') as f:
        json.dump(performance_metrics, f, indent=4)
    
    # Save predictions
    predictions_df = pd.DataFrame(all_predictions)
    predictions_path = os.path.join(output_dir, 'predictions.csv')
    predictions_df.to_csv(predictions_path, index=False)
    
    # Save predictions in JSON format as well
    predictions_json_path = os.path.join(output_dir, 'predictions.json')
    with open(predictions_json_path, 'w') as f:
        json.dump(all_predictions, f, indent=4)
    
    print(f"Inference completed. Processed {num_samples} samples in {total_time:.2f} seconds.")
    print(f"Predictions saved to {predictions_path}")
    print(f"Performance metrics saved to {performance_metrics_path}")
    
    # Return output paths
    from collections import namedtuple
    outputs = namedtuple('Outputs', ['predictions_path', 'performance_metrics_path'])
    return outputs(predictions_path=predictions_path, performance_metrics_path=performance_metrics_path)

if __name__ == '__main__':
    # For testing the component locally
    run_inference(
        data_path='/tmp/inference_data',
        model_path='/tmp/inference_model/prepared_model.pt',
        model_config_path='/tmp/inference_model/model_config.json',
        batch_size=32,
        confidence_threshold=0.5
    )
