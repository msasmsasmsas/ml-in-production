# PR2 - Labeling Tool Deployment

This PR deploys Label Studio for data annotation.

## Setup
1. Install dependencies: `pip install label-studio`
2. Run the script: `python deploy_label_studio.py`
3. Access Label Studio at `http://localhost:8080`

## Usage
- Import `data/sample_dataset.csv` from PR1.
- Create a text classification project (positive/negative labels).
- Export labeled data as CSV.