# LLM Monitoring with LangSmith

This module implements monitoring for our LLM-based agricultural threat assistant using LangSmith, providing comprehensive observability and evaluation capabilities.

## Features

- Trace and monitor LLM interactions
- Track token usage and latency
- Compare responses against expected outputs
- Evaluate model performance
- Dataset creation for continuous improvement

## Setup

1. Create a LangSmith account at [https://smith.langchain.com/](https://smith.langchain.com/)
2. Create a `.env` file with your API credentials:

```
LANGSMITH_API_KEY=your-langsmith-api-key
OPENAI_API_KEY=your-openai-api-key
LANGSMITH_PROJECT=agro_threat_assistant
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the example script to test LLM monitoring:

```bash
python langsmith_monitoring.py
```

This will create a test dataset, run several queries through the LLM, and log the results to LangSmith.

## Integration with Production System

To integrate with the production system:

1. Import the `run_with_monitoring` function
2. Pass user queries to this function instead of directly to the LLM
3. Set up feedback collection mechanisms to improve responses over time

## Dashboard Access

Access your monitoring dashboards at:
https://smith.langchain.com/project/agro_threat_assistant
