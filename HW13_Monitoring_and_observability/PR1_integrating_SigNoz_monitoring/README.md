<!-- новлена версія для PR -->
<!-- новлена версія для PR -->
# SigNoz Monitoring Integration

This directory contains the code and configuration for integrating SigNoz monitoring into our application.

## What is SigNoz?

SigNoz is an open-source APM (Application Performance Monitoring) tool that helps developers monitor their applications and troubleshoot problems. It provides traces, metrics, and logs under a single dashboard.

## Components

- `main.py`: Example application with SigNoz instrumentation
- `app/`: Application module with SigNoz tracing
- `docker-compose.yml`: Configuration for running SigNoz locally
- `kubernetes/`: Kubernetes deployment configurations

## Setup Instructions

1. Start SigNoz using Docker Compose or deploy to Kubernetes cluster
2. Configure the application with SigNoz endpoint
3. Run the application and check SigNoz dashboard for metrics

## Architecture

The application is instrumented with OpenTelemetry, which sends telemetry data to SigNoz for visualization and analysis.


