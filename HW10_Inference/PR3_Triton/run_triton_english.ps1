# Script for running Triton Inference Server in PowerShell

# Check if model repository directory exists
$MODEL_REPO_DIR = "model_repository"
if (-not (Test-Path $MODEL_REPO_DIR)) {
    Write-Host "Creating model_repository directory..."
    New-Item -Path $MODEL_REPO_DIR -ItemType Directory -Force | Out-Null
}

# Path to model repository
$MODEL_REPO = "$((Get-Location).Path -replace '\\', '/')/model_repository"

# External access ports
$HTTP_PORT = 8000
$GRPC_PORT = 8001
$METRICS_PORT = 8002

# Docker image
$DOCKER_IMAGE = "nvcr.io/nvidia/tritonserver:23.04-py3"

# Server configuration
$SERVER_ARGS = "--model-repository=/models --strict-model-config=false --log-verbose=1"

Write-Host "Starting Triton Inference Server..."
Write-Host "Using model repository: $MODEL_REPO"

# Check if Docker Desktop is running
Write-Host "Checking if Docker Desktop is running..."
try {
    $dockerInfo = docker info 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "Docker is not running"
    }
    Write-Host "Docker is running."
} catch {
    Write-Host "ERROR: Docker Desktop is not running or not installed." -ForegroundColor Red
    Write-Host "Please start Docker Desktop and try again." -ForegroundColor Yellow
    exit 1
}

# Check for GPU
$hasGPU = $false
try {
    $nvidiaSmi = nvidia-smi 2>&1
    if ($LASTEXITCODE -eq 0) {
        $hasGPU = $true
        Write-Host "GPU detected, using GPU acceleration." -ForegroundColor Green
    }
} catch {
    Write-Host "GPU not detected or drivers not installed, running without GPU acceleration." -ForegroundColor Yellow
}

# Try different volume mounting syntaxes
$success = $false

# First attempt with Windows-style path
Write-Host "Attempting to start Triton (Method 1)..." -ForegroundColor Cyan
if ($hasGPU) {
    $CMD = "docker run --gpus=all --rm -p $HTTP_PORT`:8000 -p $GRPC_PORT`:8001 -p $METRICS_PORT`:8002 -v `"$((Get-Location).Path)\model_repository`":/models $DOCKER_IMAGE tritonserver $SERVER_ARGS"
} else {
    $CMD = "docker run --rm -p $HTTP_PORT`:8000 -p $GRPC_PORT`:8001 -p $METRICS_PORT`:8002 -v `"$((Get-Location).Path)\model_repository`":/models $DOCKER_IMAGE tritonserver $SERVER_ARGS"
}

Write-Host "Running command: $CMD"
try {
    Invoke-Expression $CMD
    $success = $true
} catch {
    Write-Host "First attempt failed: $_" -ForegroundColor Yellow
}

# Second attempt with Unix-style path if first attempt failed
if (-not $success) {
    Write-Host "Attempting to start Triton (Method 2)..." -ForegroundColor Cyan
    if ($hasGPU) {
        $CMD = "docker run --gpus=all --rm -p $HTTP_PORT`:8000 -p $GRPC_PORT`:8001 -p $METRICS_PORT`:8002 -v `"$MODEL_REPO`":/models $DOCKER_IMAGE tritonserver $SERVER_ARGS"
    } else {
        $CMD = "docker run --rm -p $HTTP_PORT`:8000 -p $GRPC_PORT`:8001 -p $METRICS_PORT`:8002 -v `"$MODEL_REPO`":/models $DOCKER_IMAGE tritonserver $SERVER_ARGS"
    }

    Write-Host "Running command: $CMD"
    try {
        Invoke-Expression $CMD
        $success = $true
    } catch {
        Write-Host "Second attempt failed: $_" -ForegroundColor Yellow
    }
}

# Third attempt with alternative syntax if previous attempts failed
if (-not $success) {
    Write-Host "Attempting to start Triton (Method 3)..." -ForegroundColor Cyan
    if ($hasGPU) {
        $CMD = "docker run --gpus=all --rm -p $HTTP_PORT`:8000 -p $GRPC_PORT`:8001 -p $METRICS_PORT`:8002 -v ${PWD}/model_repository:/models $DOCKER_IMAGE tritonserver $SERVER_ARGS"
    } else {
        $CMD = "docker run --rm -p $HTTP_PORT`:8000 -p $GRPC_PORT`:8001 -p $METRICS_PORT`:8002 -v ${PWD}/model_repository:/models $DOCKER_IMAGE tritonserver $SERVER_ARGS"
    }

    Write-Host "Running command: $CMD"
    try {
        Invoke-Expression $CMD
    } catch {
        Write-Host "All attempts failed. Please check Docker installation and try again." -ForegroundColor Red
        Write-Host "See troubleshoot.md for more information." -ForegroundColor Yellow
    }
}
