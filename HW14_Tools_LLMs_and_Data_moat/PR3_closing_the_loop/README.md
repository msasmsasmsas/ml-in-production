# Closing the Loop: Creating Datasets from Monitoring

This module demonstrates how to close the machine learning feedback loop by identifying problematic instances through monitoring and creating new labeled datasets for model improvement.

## Features

- Data drift detection using Evidently
- Automated identification of problematic samples
- Creation of labeling tasks for data enrichment
- Integration with Labelbox for streamlined labeling
- Complete feedback loop implementation

## Setup

1. Create accounts with Labelbox and Arize (optional but recommended)
2. Create a `.env` file with your API credentials:

```
ARIZE_API_KEY=your-arize-api-key
ARIZE_SPACE_KEY=your-arize-space-key
LABELBOX_API_KEY=your-labelbox-api-key
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the example script to demonstrate the complete feedback loop:

```bash
python data_enrichment.py
```

This will:
1. Generate synthetic monitoring data with simulated drift
2. Detect drift and identify problematic samples
3. Create labeling tasks based on these samples
4. Upload tasks to Labelbox (if credentials are available)

## Integration with Production System

To integrate with the production system:

1. Connect your model monitoring solution (Arize, Evidently, etc.)
2. Schedule regular drift detection jobs
3. Automatically create labeling tasks for problematic instances
4. Use newly labeled data to retrain and improve your model

## Feedback Loop Process

1. **Monitoring**: Detect data drift and model performance issues
2. **Selection**: Identify problematic instances requiring attention
3. **Labeling**: Create tasks for human annotation of these instances
4. **Enrichment**: Add newly labeled data to training dataset
5. **Improvement**: Retrain models with enriched dataset
6. **Deployment**: Deploy improved model to production
7. **Repeat**: Continue the cycle for continuous improvement
