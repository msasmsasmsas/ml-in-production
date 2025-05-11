MinIO Deployment Instructions

This document provides instructions for deploying MinIO in three modes: locally, via Docker, and on Kubernetes.
Local Deployment
Requirements

    Python 3.12
    MinIO server (download from https://min.io/download)
    OS: Windows with WSL or Linux

Steps

    Download MinIO for Linux in WSL:
    bash

wget https://dl.min.io/server/minio/release/linux-amd64/minio
chmod +x minio
Create a data directory:
bash
mkdir ~/minio-data
Run the MinIO server:
bash

    ./minio server ~/minio-data --console-address ":9001"
    Open a browser at http://localhost:9001 and log in (default: minioadmin/minioadmin).

Docker Deployment
Requirements

    Docker (install in WSL: sudo apt install docker.io)

Steps

    Pull the MinIO image:
    ```

docker pull minio/minio
Run the container:
```

    docker run -p 9000:9000 -p 9001:9001 --name minio -v minio-data:/data -e "MINIO_ROOT_USER=admin" -e "MINIO_ROOT_PASSWORD=password" minio/minio server /data --console-address ":9001"
    Access the console at http://localhost:9001.

Kubernetes Deployment
Requirements

    Minikube (install in WSL: https://minikube.sigs.k8s.io/docs/start/)
    kubectl (install: sudo apt install kubectl)

Steps

    Start Minikube:
    bash

minikube start
Apply the provided minio-deployment.yaml:
bash
kubectl apply -f HW3-Data-storage-processing/pr2_minio_client/minio-deployment.yaml
Get the service URL:
bash
minikube service minio-service --url
Access the MinIO console using the provided URL.