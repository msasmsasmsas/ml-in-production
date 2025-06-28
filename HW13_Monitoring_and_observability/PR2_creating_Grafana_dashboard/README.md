<!-- новлена версія для PR -->
<!-- новлена версія для PR -->
# Grafana Dashboard for Application Monitoring

This directory contains the code and configuration for creating Grafana dashboards to monitor our application.

## Components

- `dashboards/`: JSON files for Grafana dashboards
- `prometheus/`: Prometheus configuration for metrics collection
- `docker-compose.yml`: Configuration for running Grafana and Prometheus
- `kubernetes/`: Kubernetes deployment configurations

## Setup Instructions

1. Deploy Prometheus and Grafana using Docker Compose or Kubernetes
2. Import dashboards from the dashboards directory
3. Configure data sources in Grafana to point to Prometheus

## Dashboard Features

- System metrics (CPU, memory, disk usage)
- Application metrics (request rates, latencies, error rates)
- Model performance metrics (inference time, prediction distribution)
- Data drift indicators

## Architecture

The monitoring stack consists of Prometheus for metrics collection and Grafana for visualization, with exporters to collect metrics from various components of our application.


