@echo off
REM Скрипт для запуску Triton Inference Server в Windows CMD

REM Перевірка наявності директорії моделей
IF NOT EXIST model_repository (
    echo Creating model_repository directory...
    mkdir model_repository
)

REM Порти для зовнішнього доступу
SET HTTP_PORT=8000
SET GRPC_PORT=8001
SET METRICS_PORT=8002

REM Образ Docker
SET DOCKER_IMAGE=nvcr.io/nvidia/tritonserver:23.04-py3

REM Конфігурація сервера
SET SERVER_ARGS=--model-repository=/models --strict-model-config=false --log-verbose=1

echo Starting Triton Inference Server...

REM Спроба запуску без GPU опції спочатку
echo Trying to start without GPU option...
docker run --rm -p %HTTP_PORT%:8000 -p %GRPC_PORT%:8001 -p %METRICS_PORT%:8002 -v %cd%/model_repository:/models %DOCKER_IMAGE% tritonserver %SERVER_ARGS%

echo If startup failed, try running with GPU option:
echo docker run --gpus=all --rm -p %HTTP_PORT%:8000 -p %GRPC_PORT%:8001 -p %METRICS_PORT%:8002 -v %cd%/model_repository:/models %DOCKER_IMAGE% tritonserver %SERVER_ARGS%

pause
