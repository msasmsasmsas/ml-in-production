# Скрипт для запуску Triton Inference Server в PowerShell

# Перевірка наявності директорії моделей
$MODEL_REPO_DIR = "model_repository"
if (-not (Test-Path $MODEL_REPO_DIR)) {
    Write-Host "Створення директорії model_repository..."
    New-Item -Path $MODEL_REPO_DIR -ItemType Directory -Force | Out-Null
}

# Шлях до репозиторію моделей
$MODEL_REPO = "$((Get-Location).Path -replace '\\', '/')/model_repository"

# Порти для зовнішнього доступу
$HTTP_PORT = 8000
$GRPC_PORT = 8001
$METRICS_PORT = 8002

# Образ Docker
$DOCKER_IMAGE = "nvcr.io/nvidia/tritonserver:23.04-py3"

# Конфігурація сервера
$SERVER_ARGS = "--model-repository=/models --strict-model-config=false --log-verbose=1"

Write-Host "Запуск Triton Inference Server..."
Write-Host "Використання репозиторію моделей: $MODEL_REPO"

# Перевірка наявності GPU
$hasGPU = $false
try {
    $nvidiaSmi = Invoke-Expression "nvidia-smi" -ErrorAction SilentlyContinue
    if ($nvidiaSmi -ne $null) {
        $hasGPU = $true
        Write-Host "GPU знайдено, використовуємо GPU-прискорення."
    }
} catch {
    Write-Host "GPU не знайдено або драйвери не встановлені, запускаємо без GPU-прискорення."
}

# Формування та виконання команди
if ($hasGPU) {
    $CMD = "docker run --gpus=all --rm -p $HTTP_PORT`:8000 -p $GRPC_PORT`:8001 -p $METRICS_PORT`:8002 -v `"$MODEL_REPO`":/models $DOCKER_IMAGE tritonserver $SERVER_ARGS"
} else {
    $CMD = "docker run --rm -p $HTTP_PORT`:8000 -p $GRPC_PORT`:8001 -p $METRICS_PORT`:8002 -v `"$MODEL_REPO`":/models $DOCKER_IMAGE tritonserver $SERVER_ARGS"
}

Write-Host "Виконання команди: $CMD"
Invoke-Expression $CMD
