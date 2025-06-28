@echo off
REM Script for running Triton Inference Server in Windows CMD

REM Check if model repository directory exists
IF NOT EXIST model_repository (
    echo Creating model_repository directory...
    mkdir model_repository
)

REM External access ports
SET HTTP_PORT=8000
SET GRPC_PORT=8001
SET METRICS_PORT=8002

REM Docker image
SET DOCKER_IMAGE=nvcr.io/nvidia/tritonserver:23.04-py3

REM Server configuration
SET SERVER_ARGS=--model-repository=/models --strict-model-config=false --log-verbose=1

echo Starting Triton Inference Server...

REM Try running without GPU option first
echo Trying to start without GPU option...

REM Check if Docker Desktop is running
echo Checking if Docker Desktop is running...
docker info > nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo ERROR: Docker Desktop is not running or not installed.
    echo Please start Docker Desktop and try again.
    goto END
)

REM Run Triton server without GPU
docker run --rm -p %HTTP_PORT%:8000 -p %GRPC_PORT%:8001 -p %METRICS_PORT%:8002 -v "%cd%\model_repository":/models %DOCKER_IMAGE% tritonserver %SERVER_ARGS%

IF %ERRORLEVEL% NEQ 0 (
    echo First attempt failed. Trying alternate path format...
    docker run --rm -p %HTTP_PORT%:8000 -p %GRPC_PORT%:8001 -p %METRICS_PORT%:8002 -v %cd%/model_repository:/models %DOCKER_IMAGE% tritonserver %SERVER_ARGS%
)

echo.
echo If both attempts failed, try running with GPU option:
echo docker run --gpus=all --rm -p %HTTP_PORT%:8000 -p %GRPC_PORT%:8001 -p %METRICS_PORT%:8002 -v "%cd%\model_repository":/models %DOCKER_IMAGE% tritonserver %SERVER_ARGS%

:END
pause
