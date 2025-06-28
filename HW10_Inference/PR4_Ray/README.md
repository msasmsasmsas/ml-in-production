св# Ray Serve Model Deployment
# Ray Serve для розгортання моделі класифікації зображень

Цей проект демонструє розгортання моделі класифікації зображень за допомогою Ray Serve.

## Передумови

- Python 3.9+
- Ray Serve 2.7.0
- Встановлені пакети з requirements.txt

## Структура проєкту

- `app/` - основний код додатку
  - `models/` - код моделі класифікації зображень
  - `deployments/` - код розгортання Ray Serve
  - `main.py` - головний файл запуску
- `tests/` - тести
- `client/` - клієнтський код для тестування API

## Встановлення залежностей

### Спосіб 1: Використання скриптів встановлення

**На Windows:**
```powershell
.\install.ps1
```

**На Linux/macOS:**
```bash
chmod +x install.sh
./install.sh
```

### Спосіб 2: Ручне встановлення

```bash
# Створення віртуального середовища
python -m venv venv

# Активація середовища
# На Windows:
venv\Scripts\activate
# На Linux/macOS:
source venv/bin/activate

# Встановлення залежностей окремо, щоб уникнути конфліктів
pip install fastapi==0.103.1 uvicorn==0.23.2 python-multipart==0.0.6
pip install pillow==10.0.0 numpy==1.24.3 pydantic==1.10.12
pip install torch==2.0.1 torchvision==0.15.2
pip install ray[serve]==2.6.1
```

## Запуск

### Спосіб 1: Використання скриптів запуску

**На Windows:**
```powershell
.\run.ps1
```

**На Linux/macOS:**
```bash
chmod +x run.sh
./run.sh
```

### Спосіб 2: Запуск з кореня проєкту

```bash
python -m app.main
```

### Спосіб 3: Запуск з директорії проєкту

```bash
cd HW10_Inference/PR4_Ray
python -m app.main
```

## Тестування API

Після запуску сервісу, ви можете перевірити його роботу за допомогою HTTP запитів:

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@path/to/image.jpg" \
  -F "top_k=5"
```

Або використовуючи клієнтський код з директорії `client/`.

## Моніторинг

Робота Ray Serve доступна через дашборд за адресою: http://localhost:8265
This directory contains code for deploying a machine learning model using Ray Serve.

## Overview

Ray Serve is a scalable, framework-agnostic model serving library that enables you to serve everything from deep learning models to business logic with horizontal scalability.

## Structure

- `app/` - Contains the Ray Serve application code
  - `deployments/` - Model deployment definitions
  - `models/` - ML model code
- `client/` - Client code for sending requests to the Ray Serve deployment
- `tests/` - Tests for the deployment
- `requirements.txt` - Required dependencies

## Setup

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Running the server

```bash
python app/main.py
```

## Running the client
# Ray Serve для інференсу моделей машинного навчання
# Ray Serve для розгортання моделі класифікації зображень
# Ray Serve для розгортання моделі класифікації зображень

## Запуск

Для запуску використовуйте команду:

```bash
python -m app.main --no-uvicorn
```

## Використання API

Після запуску сервера, API буде доступний за адресою `http://localhost:8000/`.

### Ендпоінти

- `GET /health` - Перевірка стану сервера
- `GET /metadata` - Отримання метаданих моделі
- `POST /predict` - Класифікація зображення
Цей проект демонструє використання Ray Serve для розгортання моделі класифікації зображень на основі ResNet50.

## Швидкий старт

### Спрощений запуск (рекомендується)

```powershell
# Для Windows
.\run_server.ps1
```

```bash
# Для Linux/macOS
chmod +x run_server.sh
./run_server.sh
```

Ці скрипти виправляють файл JSON з класами та запускають Ray Serve без uvicorn, що вирішує проблеми з імпортом модулів.

### Альтернативний метод запуску

```powershell
# Для Windows
python -m app.main --no-uvicorn
```

```bash
# Для Linux/macOS
python -m app.main --no-uvicorn
```

## Структура проекту

- `app/` - Основний код додатку
  - `deployments/` - Розгортання Ray Serve
    - `classifier.py` - Розгортання для класифікації зображень
    - `router.py` - Розгортання для маршрутизації HTTP-запитів
  - `models/` - Моделі машинного навчання
    - `classifier.py` - Класифікатор зображень на основі ResNet50
    - `imagenet_classes.json` - Класи ImageNet
  - `main.py` - Головний файл входу
- `client/` - Клієнтський код для взаємодії з API
- `tests/` - Тести

## API Endpoints

- `POST /predict` - Класифікувати зображення
  - Параметри: `file` (зображення), `top_k` (кількість прогнозів)
- `GET /health` - Перевірка стану
- `GET /metadata` - Метадані сервісу

## Вимоги

- Python 3.9+
- Ray Serve 2.6.1+
- PyTorch
- FastAPI
- PIL (Pillow)
- NumPy

## Вирішення проблем

Якщо у вас виникають проблеми з запуском:

1. Переконайтеся, що ви використовуєте правильну версію Ray (2.6.1 рекомендується)
2. Перевірте, що порт 8000 вільний
3. Використовуйте ізольоване віртуальне середовище
Цей проект демонструє використання Ray Serve для розгортання моделей класифікації зображень.

## Структура проекту

- `app/` - Основний код додатку
  - `models/` - Моделі машинного навчання
  - `deployments/` - Ray Serve розгортання
  - `main.py` - Точка входу додатку
- `client/` - Клієнтські скрипти для тестування API
- `tests/` - Модульні та інтеграційні тести

## Передумови

- Python 3.9+
- PyTorch
- Ray та Ray Serve
- FastAPI
- Uvicorn

## Встановлення

```bash
# Встановлення залежностей
pip install -r requirements.txt
```

## Запуск сервісу

```bash
# Запуск з кореневої директорії проекту
python -m app.main
```

Або можна використати uvicorn напряму:

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

## API Ендпоінти

- `POST /predict` - Класифікація зображення
- `GET /health` - Перевірка стану сервісу
- `GET /metadata` - Отримання метаданих моделі

## Приклад використання

```python
import requests
import json
from PIL import Image
import io

# Завантаження зображення
image_path = "path/to/image.jpg"
image = Image.open(image_path)

# Конвертація зображення у байти
buffer = io.BytesIO()
image.save(buffer, format="JPEG")
image_bytes = buffer.getvalue()

# Відправка запиту на класифікацію
files = {
    'file': ('image.jpg', image_bytes, 'image/jpeg')
}
response = requests.post(
    'http://localhost:8000/predict',
    files=files,
    params={'top_k': 5}
)

# Виведення результатів
print(json.dumps(response.json(), indent=2))
```

## Вирішення проблем

### Проблема з імпортами

Якщо виникає помилка з імпортами, переконайтеся, що запускаєте додаток з кореневої директорії проекту:

```bash
# Правильно
cd HW10_Inference/PR4_Ray
python -m app.main

# Неправильно
cd HW10_Inference/PR4_Ray/app
python main.py
```

### Помилка підключення до Ray

Якщо виникає помилка з підключенням до Ray, переконайтеся, що Ray не запущено в іншому процесі:

```bash
# Перевірка запущених процесів Ray
ps aux | grep ray

# Зупинка всіх процесів Ray
ray stop
```
```bash
python client/client.py --image tests/test_image.jpg
```

## Running tests

```bash
pytest tests/
```

## Features

- Scalable ML model serving with Ray Serve
- Support for multiple models and model versions
- Batching requests for higher throughput
- Monitoring and metrics with Prometheus integration
- Fault tolerance and automatic scaling
