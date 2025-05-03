# HW2: Infrastructure Setup

This project implements the infrastructure setup for the Machine Learning in Production course.

## PR1: FastAPI Docker Setup
- **Description**: Created a Dockerfile for a FastAPI server.
- **Files**:
  - `Dockerfile`: Configures the FastAPI environment.
  - `app/main.py`: FastAPI app with `/` endpoint.
  - `app/requirements.txt`: Lists `fastapi`, `uvicorn`.
- **Run**:
  ```bash
  docker build -t msasmsasmsas/fastapi-server:latest .
  docker run -p 8000:8000 msasmsasmsas/fastapi-server:latest