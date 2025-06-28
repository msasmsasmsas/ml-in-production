# новлена версія для PR
FROM python:3.10-slim

WORKDIR /app

# Р’СЃС‚Р°РЅРѕРІР»СЋС”РјРѕ Р·Р°Р»РµР¶РЅРѕСЃС‚С–
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# РљРѕРїС–СЋС”РјРѕ РєРѕРґ РґРѕРґР°С‚РєСѓ
COPY . .

# Р’СЃС‚Р°РЅРѕРІР»СЋС”РјРѕ Р·РјС–РЅРЅС– СЃРµСЂРµРґРѕРІРёС‰Р°
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    OTEL_ENABLED=true \
    OTEL_SERVICE_NAME=threat-detection-api \
    OTEL_EXPORTER_OTLP_ENDPOINT=http://signoz-otel-collector:4317

# Р—Р°РїСѓСЃРєР°С”РјРѕ РґРѕРґР°С‚РѕРє
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

