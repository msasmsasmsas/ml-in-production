# Triton Inference Server Deployment
# Triton Inference Server - Клієнт для моделі ResNet50

Цей проект демонструє використання Triton Inference Server для інференсу моделі ResNet50.

## Структура проекту

- `client/` - Клієнтський код для взаємодії з Triton Inference Server
  - `client.py` - Основний скрипт клієнта для відправки запитів
  - `imagenet_classes.json` - Повний словник класів ImageNet
- `tests/` - Тестові зображення
- `download_image.py` - Скрипт для завантаження тестових зображень

## Передумови

- Python 3.9+
- Triton Inference Server (запущений локально або в Docker)
- Модель ResNet50, завантажена в Triton Server

## Встановлення

```bash
# Встановлення залежностей
pip install tritonclient[http] pillow numpy requests

# Завантаження тестового зображення
python download_image.py
```

## Використання

### 1. Запуск клієнта

```bash
python client/client.py --image tests/test_image.jpg --model resnet50
```

Додаткові опції:

- `--url` - URL-адреса Triton сервера (за замовчуванням: localhost:8000)
- `--verbose` - Включити детальне логування

### 2. Перевірка статусу моделі

```bash
curl -v localhost:8000/v2/models/resnet50/ready
```

## Вирішення проблем

1. **Помилка з'єднання** - Переконайтеся, що Triton Server запущений і доступний за вказаною URL-адресою.

2. **Модель не готова** - Перевірте, чи правильно завантажена модель в Triton Server.

3. **Помилка типу даних** - Клієнт налаштований на роботу з даними типу FP32. Переконайтеся, що модель очікує такий самий тип даних.

## Результати класифікації

Клієнт поверне топ-5 прогнозів для вхідного зображення в форматі:

```
Топ-5 прогнозів:
1. Samoyed: 13.0685
2. Pomeranian: 9.7074
3. keeshond: 8.8239
4. Great Pyrenees: 8.2490
5. Japanese spaniel: 7.4462
```

Це показує найбільш вірогідні класи для зображення та їх бали.
This directory contains code for deploying a machine learning model using NVIDIA Triton Inference Server.

## Overview

Triton Inference Server provides a cloud and edge inferencing solution optimized for both CPUs and GPUs. It supports multiple frameworks including TensorFlow, PyTorch, ONNX, and custom backends.

## Structure

- `model_repository/` - The model repository structure required by Triton
- `client/` - Client code for sending inference requests
- `tests/` - Tests for the deployment
- `requirements.txt` - Required dependencies

## Setup

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Running the server

### Windows PowerShell

```powershell
# PowerShell - варіант без GPU (рекомендується спочатку спробувати без GPU)
docker run --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 -v "${PWD}/model_repository":/models nvcr.io/nvidia/tritonserver:23.04-py3 tritonserver --model-repository=/models --strict-model-config=false

# PowerShell - варіант з GPU
docker run --gpus=all --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 -v "${PWD}/model_repository":/models nvcr.io/nvidia/tritonserver:23.04-py3 tritonserver --model-repository=/models --strict-model-config=false

# PowerShell - альтернативний варіант для шляхів з пробілами
docker run --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 -v "$((Get-Location).Path -replace '\\', '/')/model_repository":/models nvcr.io/nvidia/tritonserver:23.04-py3 tritonserver --model-repository=/models --strict-model-config=false
```

### Windows CMD

```cmd
# Без GPU опції (рекомендується спочатку спробувати без GPU)
docker run --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 -v %cd%/model_repository:/models nvcr.io/nvidia/tritonserver:23.04-py3 tritonserver --model-repository=/models --strict-model-config=false

# З GPU опцією
docker run --gpus=all --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 -v %cd%/model_repository:/models nvcr.io/nvidia/tritonserver:23.04-py3 tritonserver --model-repository=/models --strict-model-config=false
```

### Linux/MacOS

```bash
docker run --gpus=all --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 -v $(pwd)/model_repository:/models nvcr.io/nvidia/tritonserver:23.04-py3 tritonserver --model-repository=/models
```

## Running the client

```bash
python client/client.py
```

## Running tests

```bash
pytest tests/
```

## Вирішення проблем

Якщо у вас виникли проблеми з запуском Triton сервера (зависання під час завантаження, SIGTERM/SIGINT помилки), спробуйте наступні кроки:

### 1. Перевірте наявність директорії моделей

```powershell
# Створіть директорію, якщо вона не існує
mkdir -p model_repository/resnet50/1
```

### 2. Перевірте наявність GPU та налаштування NVIDIA Container Toolkit

```powershell
# Перевірка GPU
nvidia-smi

# Перевірка налаштувань Docker
docker info | Select-String "Runtimes"
```

### 3. Спробуйте спочатку завантажити образ Docker окремо

```powershell
# Спочатку завантажте образ
docker pull nvcr.io/nvidia/tritonserver:23.04-py3
```

### 4. Запустіть з додатковими параметрами

```powershell
# Запуск без GPU та з відключеною перевіркою конфігурації моделей
docker run --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 -v "${PWD}/model_repository":/models nvcr.io/nvidia/tritonserver:23.04-py3 tritonserver --model-repository=/models --strict-model-config=false --log-verbose=1
```

### 5. Використовуйте готові скрипти

```powershell
# Використовуйте готовий скрипт PowerShell
./run_triton.ps1

# Або скрипт для CMD
run_triton.bat
```

### 6. Перевірте наявність моделі та її конфігурації

Переконайтеся, що в директорії `model_repository/resnet50` є правильна конфігурація та модель:

```
model_repository/
└── resnet50/
    ├── config.pbtxt    # Конфігурація моделі
    └── 1/              # Версія моделі
        └── model.onnx  # Файл моделі (якщо відсутній, використовуйте model_converter.py)
```

### 7. Конвертуйте модель у формат ONNX, якщо потрібно

```powershell
# Конвертація PyTorch моделі в ONNX формат
python model_converter.py --model resnet50 --output model_repository/resnet50/1/model.onnx --create_config
```
