FROM python:3.10-slim

WORKDIR /app

# Встановлюємо залежності
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копіюємо код додатку
COPY . .

# Встановлюємо змінні середовища
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    OTEL_ENABLED=true \
    OTEL_SERVICE_NAME=threat-detection-api \
    OTEL_EXPORTER_OTLP_ENDPOINT=http://signoz-otel-collector:4317

# Запускаємо додаток
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
