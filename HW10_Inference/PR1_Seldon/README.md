# Розгортання моделі ResNet50 з використанням Seldon Core

Цей проект демонструє розгортання моделі класифікації зображень ResNet50 за допомогою Seldon Core.

## Структура проекту

- `model/` - код моделі ResNet50Classifier
- `Dockerfile` - основний Dockerfile для побудови контейнера
- `Dockerfile.simplified` - альтернативний Dockerfile з детальними кроками
- `requirements.txt` - залежності Python
- `kubernetes/` - YAML-файли для розгортання в Kubernetes

## Передумови

- Python 3.9+
- Docker
- kubectl (для розгортання в Kubernetes)
- Доступ до кластера Kubernetes (для розгортання в Kubernetes)

## Варіанти розгортання

### 1. Локальне розгортання

Для локального запуску та тестування моделі без контейнеризації:

```bash
# Встановлення залежностей
pip install -r requirements.txt

# Запуск сервісу
python -m seldon_core.microservice model.ResNet50Classifier --service-type MODEL
```

Це запустить сервіс на порту 9000 (за замовчуванням).

### 2. Розгортання з використанням Docker

```bash
# Збірка Docker-образу
docker build -t resnet50-classifier:latest .

# Запуск контейнера
docker run -p 9000:9000 resnet50-classifier:latest
```

### 3. Розгортання в Kubernetes

```bash
# Створення Docker-образу та відправлення в реєстр
docker build -t your-registry/resnet50-classifier:latest .
docker push your-registry/resnet50-classifier:latest

# Розгортання в Kubernetes
kubectl apply -f kubernetes/deployment.yaml
kubectl apply -f kubernetes/service.yaml
```

Якщо ви використовуєте Seldon Core Operator у вашому кластері Kubernetes:

```bash
# Розгортання SeldonDeployment
kubectl apply -f kubernetes/seldon-deployment.yaml
```

## Тестування API

Після розгортання (будь-яким методом) ви можете протестувати API:

```bash
curl -X POST http://localhost:9000/api/v1.0/predictions \
   -H 'Content-Type: application/json' \
   -d '{"data": {"ndarray": [[[[...image data...]]]]}}'
```

Також можна використовувати Python-клієнт:

```python
import requests
import numpy as np
from PIL import Image
import base64
import io

# Завантаження та кодування зображення
image = Image.open('path/to/image.jpg')
buffered = io.BytesIO()
image.save(buffered, format="JPEG")
img_str = base64.b64encode(buffered.getvalue()).decode()

# Відправлення запиту
response = requests.post(
    'http://localhost:9000/api/v1.0/predictions',
    json={"strData": img_str}
)
print(response.json())
```

## Додаткова інформація

Документація Seldon Core: [https://docs.seldon.io/](https://docs.seldon.io/)
