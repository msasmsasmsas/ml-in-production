#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import NamedTuple
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from google.cloud import storage
import shutil
from datetime import datetime

def save_results(
    predictions_path: str, 
    output_path: str,
    create_visualizations: bool = True
) -> NamedTuple('Outputs', [('results_output_path', str)]):
    """
    Save inference results to the specified output path
    
    Args:
        predictions_path: Path to the inference predictions
        output_path: Path to save the results
        create_visualizations: Whether to create visualizations of the results
        
    Returns:
        results_output_path: Path where the results were saved
    """
    print(f"Saving inference results from {predictions_path} to {output_path}")
    
    # Create a timestamp-based folder name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_folder = f"inference_results_{timestamp}"
    
    # Create a local directory to organize the files
    local_output_dir = f"/tmp/final_results/{results_folder}"
    os.makedirs(local_output_dir, exist_ok=True)
    
    # Load predictions
    predictions_df = pd.read_csv(predictions_path)
    
    # Copy predictions file
    shutil.copy(predictions_path, os.path.join(local_output_dir, 'predictions.csv'))
    
    # Load performance metrics if available
    perf_metrics_path = os.path.join(os.path.dirname(predictions_path), 'performance_metrics.json')
    if os.path.exists(perf_metrics_path):
        with open(perf_metrics_path, 'r') as f:
            performance_metrics = json.load(f)
        
        # Copy performance metrics
        shutil.copy(perf_metrics_path, os.path.join(local_output_dir, 'performance_metrics.json'))
    else:
        performance_metrics = {}
    
    # Create visualizations if requested
    if create_visualizations:
        # Create visualizations directory
        viz_dir = os.path.join(local_output_dir, 'visualizations')
        os.makedirs(viz_dir, exist_ok=True)
        
        # 1. Confidence distribution
        if 'confidence' in predictions_df.columns:
            plt.figure(figsize=(10, 6))
            sns.histplot(predictions_df['confidence'], bins=20)
            plt.title('Distribution of Prediction Confidence')
            plt.xlabel('Confidence Score')
            plt.ylabel('Count')
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, 'confidence_distribution.png'))
            plt.close()
        
        # 2. Top predicted classes
        if 'predicted_class' in predictions_df.columns:
            plt.figure(figsize=(12, 8))
            top_classes = predictions_df['predicted_class'].value_counts().head(10)
            sns.barplot(x=top_classes.values, y=top_classes.index)
            plt.title('Top 10 Predicted Classes')
            plt.xlabel('Count')
            plt.ylabel('Class')
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, 'top_predicted_classes.png'))
            plt.close()
        
        # 3. Confusion matrix if true labels are available
        if 'true_class' in predictions_df.columns and 'predicted_class' in predictions_df.columns:
            from sklearn.metrics import confusion_matrix
            
            # Get unique classes
            classes = sorted(list(set(predictions_df['true_class'].unique()) | 
                                 set(predictions_df['predicted_class'].unique())))
            
            # Create confusion matrix
            cm = confusion_matrix(
                predictions_df['true_class'], 
                predictions_df['predicted_class'],
                labels=classes
            )
            
            # Plot confusion matrix
            plt.figure(figsize=(12, 10))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=classes, yticklabels=classes)
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title('Confusion Matrix')
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, 'confusion_matrix.png'))
            plt.close()
            
            # 4. Create classification report
            from sklearn.metrics import classification_report
            
            report = classification_report(
                predictions_df['true_class'],
                predictions_df['predicted_class'],
                output_dict=True
            )
            
            # Save classification report
            with open(os.path.join(viz_dir, 'classification_report.json'), 'w') as f:
                json.dump(report, f, indent=4)
            
            # Create classification metrics summary
            metrics_summary = {
                'accuracy': report['accuracy'],
                'macro_avg_precision': report['macro avg']['precision'],
                'macro_avg_recall': report['macro avg']['recall'],
                'macro_avg_f1': report['macro avg']['f1-score'],
                'weighted_avg_precision': report['weighted avg']['precision'],
                'weighted_avg_recall': report['weighted avg']['recall'],
                'weighted_avg_f1': report['weighted avg']['f1-score']
            }
            
            with open(os.path.join(local_output_dir, 'metrics_summary.json'), 'w') as f:
                json.dump(metrics_summary, f, indent=4)
    
    # Create a summary file
    summary = {
        'timestamp': timestamp,
        'num_predictions': len(predictions_df),
        'performance_metrics': performance_metrics,
        'visualizations_created': create_visualizations
    }
    
    with open(os.path.join(local_output_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=4)
    
    # If output_path is a GCS path, upload the files
    if output_path.startswith('gs://'):
        # Parse the bucket and blob prefix
        bucket_name = output_path.split('gs://')[1].split('/')[0]
        prefix = '/'.join(output_path.split('gs://')[1].split('/')[1:])
        results_prefix = f"{prefix}/{results_folder}" if prefix else results_folder
        
        # Initialize GCS client
        storage_client = storage.Client()
        bucket = storage_client.get_bucket(bucket_name)
        
        # Upload all files in the local directory
        for root, dirs, files in os.walk(local_output_dir):
            for file in files:
                local_file = os.path.join(root, file)
                relative_path = os.path.relpath(local_file, local_output_dir)
                blob_name = f"{results_prefix}/{relative_path}"
                
                blob = bucket.blob(blob_name)
                blob.upload_from_filename(local_file)
                print(f"Uploaded {local_file} to gs://{bucket_name}/{blob_name}")
        
        # Set the results output path to the GCS path
        results_output_path = f"gs://{bucket_name}/{results_prefix}"
    else:
        # If it's a local path, create the directory and copy the files
        results_dir = os.path.join(output_path, results_folder)
        os.makedirs(results_dir, exist_ok=True)
        
        # Copy all files from the temporary directory
        for root, dirs, files in os.walk(local_output_dir):
            for file in files:
                src_file = os.path.join(root, file)
                relative_path = os.path.relpath(src_file, local_output_dir)
                dst_file = os.path.join(results_dir, relative_path)
                
                # Create destination directory if it doesn't exist
                os.makedirs(os.path.dirname(dst_file), exist_ok=True)
                
                # Copy the file
                shutil.copyfile(src_file, dst_file)
                print(f"Copied {src_file} to {dst_file}")
        
        # Set the results output path to the local path
        results_output_path = results_dir
    
    print(f"Inference results saved to {results_output_path}")
    
    # Return the results output path
    from collections import namedtuple
    outputs = namedtuple('Outputs', ['results_output_path'])
    return outputs(results_output_path=results_output_path)

if __name__ == '__main__':
    # For testing the component locally
    save_results(
        predictions_path='/tmp/inference_results/predictions.csv',
        output_path='./saved_results',
        create_visualizations=True
    )
