FROM python:3.9-slim

# Встановлення Git та базових залежностей
RUN apt-get update && \
    apt-get install -y git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Встановлення залежностей
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копіювання файлів моделі
COPY model/ ./model/

# Створення файлу __init__.py для коректного імпорту модуля
RUN touch /app/model/__init__.py

# Налаштування змінних середовища
ENV MODEL_NAME="ResNet50Classifier"
ENV SERVICE_TYPE="MODEL"
ENV PERSISTENCE="0"
ENV PYTHONPATH="${PYTHONPATH}:/app"

# Відкриття порту
EXPOSE 9000

# Команда запуску
CMD ["python", "-m", "seldon_core.microservice", "model.ResNet50Classifier", "--service-type", "MODEL"]
