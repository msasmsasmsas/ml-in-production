# HW2: Infrastructure Setup

This project implements the infrastructure setup for the Machine Learning in Production course.

## PR1: FastAPI Docker Setup
- **Description**: Created a Dockerfile for a FastAPI server.
- **Files**:
  - `Dockerfile`: Configures the FastAPI environment.
<<<<<<< HEAD
  - `app/main.py`: FastAPI app with `/` endpoint.
=======
  - `app/main.py`: FastAPI app with `/` and `/health` endpoints.
>>>>>>> new-pr2-cicd
  - `app/requirements.txt`: Lists `fastapi`, `uvicorn`.
- **Run**:
  ```bash
  docker build -t msasmsas/fastapi-server:latest .
  docker run -p 8000:8000 msasmsasmsas/fastapi-server:latest
<<<<<<< HEAD
  
Result: Image built and pushed to Docker Hub. Server accessible at http://localhost:8000.
=======

    Result: Image built and pushed to Docker Hub. Server accessible at http://localhost:8000 and http://localhost:8000/health.
>>>>>>> new-pr2-cicd

PR2: CI/CD Pipeline

    Description: Added GitHub Actions pipeline to build and push Docker image.
    Files:
        .github/workflows/ci-cd.yml: CI/CD pipeline.
    Run:
        Runs on PRs and pushes to main.
        Status: https://github.com/msasmsasmsas/ml-in-production/actions.
    Result: Pipeline builds and pushes msasmsasmsas/fastapi-server:latest. All runs are green.

PR3: Kubernetes Deployment

    Description: Created Kubernetes manifests, tested with Minikube.
    Files:
        k8s/pod.yaml: Single Pod.
        k8s/deployment.yaml: Deployment with 3 replicas.
        k8s/service.yaml: ClusterIP Service.
        k8s/job.yaml: One-time Job.
    Run:
    bash

    minikube start
    kubectl apply -f k8s/
    minikube service fastapi-service
    Result: Resources deployed. Service accessible via Minikube.

CI/CD Results

    Pipeline runs on PRs and pushes to main.
    Builds and pushes msasmsasmsas/fastapi-server:latest.
    Runs are green: https://github.com/msasmsasmsas/ml-in-production/actions.

Setup

    Clone:
    bash

git clone https://github.com/msasmsasmsas/ml-in-production.git
cd ml-in-production/HW2-Infrastructure-setup
Run Docker:
bash
docker build -t msasmsasmsas/fastapi-server:latest .
docker run -p 8000:8000 msasmsasmsas/fastapi-server:latest
Deploy to Kubernetes:
bash
minikube start
kubectl apply -f k8s/
minikube service fastapi-service
Install k9s:
bash

    choco install k9s
    k9s

Conclusion

<<<<<<< HEAD
PR1, PR2, and PR3 are completed. PRs are created and merged. CI/CD is green, Kubernetes tested.
text

Закоммітіть у гілку `new-pr2-cicd`:

```powershell
cd E:\ml-in-production\HW2-Infrastructure-setup
git add README.md
git commit -m "Add README for HW2"
git push origin new-pr2-cicd

Потім перенесіть README.md до PR3 (якщо потрібно):
powershell
git checkout new-pr3-kubernetes
git merge new-pr2-cicd
git push origin new-pr3-kubernetes
=======
PR1, PR2, and PR3 are completed. PRs are created and merged. CI/CD is green, Kubernetes tested.
>>>>>>> new-pr2-cicd
