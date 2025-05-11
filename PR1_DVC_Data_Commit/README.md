# PR1 - DVC Data Commit

This PR commits a sample dataset using DVC.

## Dataset Description
- File: `data/sample_dataset.csv`
- Columns: id, text, label
- Size: ~50 samples (based on manual labeling experience)
- Version: v1.0

## Labeling Cost and Time Estimation
- **Time**: ~2 hours for 50 samples (2.4 minutes per sample).
- **Cost**: Assuming $15/hour, total cost is ~$30 for 50 samples.

## Labeling Instructions
1. Read the text column.
2. Assign a label (positive/negative) based on sentiment.
3. Save annotations in CSV format.

## Data Enrichment Flow in Production
- **Step 1**: Collect raw text data (e.g., via web scraping).
- **Step 2**: Preprocess data (remove duplicates, normalize text).
- **Step 3**: Label data using Label Studio (see PR2).
- **Step 4**: Validate labels with Cleanlab (see PR4).
- **Step 5**: Store enriched data in DVC for versioning.

## Google Doc
[Link to Google Doc](# [Link to Google Doc](https://docs.google.com/document/d/1wIEtSHri2KdLumByjjXU53WLuqK94M_WC0xkwnyimy8)