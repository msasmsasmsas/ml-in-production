#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Головний файл додатку з інтеграцією моніторингу SigNoz через OpenTelemetry
"""

import os
import sys
import time
import logging
import traceback
from fastapi import FastAPI, HTTPException, Depends, File, UploadFile, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import io
import uuid
from contextlib import asynccontextmanager

# Імпортуємо модулі для SigNoz/OpenTelemetry
from app.telemetry import setup_telemetry
from app.model import ThreatDetectionModel
from app.schemas import PredictionResponse, PredictionRequest
from app.config import settings

# Налаштування логування
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Ініціалізуємо телеметрію
setup_telemetry(service_name="threat-detection-api")

# Створюємо контекстний менеджер для запуску та зупинки сервісів
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Виконується при старті
    logger.info("Ініціалізація додатку...")
    # Ініціалізуємо модель при старті
    app.state.model = ThreatDetectionModel()
    yield
    # Виконується при зупинці
    logger.info("Завершення роботи додатку...")

# Створюємо FastAPI додаток
app = FastAPI(
    title="Threat Detection API",
    description="API для виявлення загроз сільськогосподарським культурам",
    version="1.0.0",
    lifespan=lifespan
)

# Додаємо CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """
    Кореневий ендпоінт для перевірки працездатності API
    """
    return {"status": "ok", "message": "Threat Detection API працює"}

@app.get("/health")
async def health_check():
    """
    Ендпоінт для перевірки здоров'я сервісу
    """
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "version": settings.API_VERSION,
        "model_loaded": hasattr(app.state, "model") and app.state.model is not None
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(background_tasks: BackgroundTasks, file: UploadFile = File(...), threshold: float = 0.5):
    """
    Ендпоінт для передбачення загроз на зображенні

    - **file**: зображення для аналізу
    - **threshold**: поріг впевненості (0.0-1.0)
    """
    request_id = str(uuid.uuid4())
    logger.info(f"Отримано запит на передбачення [ID: {request_id}]")

    # Перевіряємо, чи це зображення
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Файл повинен бути зображенням")

    try:
        # Читаємо зображення
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))

        # Робимо передбачення
        model = app.state.model
        result = model.predict(image, confidence_threshold=threshold)

        # Формуємо відповідь
        response = PredictionResponse(
            request_id=request_id,
            threats=result["threats"],
            recommendations=result["recommendations"],
            details=result["details"]
        )

        # Додаємо асинхронне завдання для логування результатів
        background_tasks.add_task(log_prediction_result, request_id, len(result["threats"]))

        return response

    except Exception as e:
        logger.error(f"Помилка при обробці зображення: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Помилка обробки: {str(e)}")

def log_prediction_result(request_id: str, threat_count: int):
    """
    Функція для асинхронного логування результатів передбачення
    """
    logger.info(f"Завершено передбачення [ID: {request_id}], виявлено загроз: {threat_count}")

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """
    Глобальний обробник винятків
    """
    logger.error(f"Непередбачена помилка: {str(exc)}")
    logger.error(traceback.format_exc())
    return JSONResponse(
        status_code=500,
        content={"detail": "Виникла внутрішня помилка сервера"}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
