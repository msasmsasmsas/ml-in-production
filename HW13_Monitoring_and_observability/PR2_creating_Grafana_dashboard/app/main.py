#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Головний файл додатку з інтеграцією Prometheus для моніторингу
"""

import os
import sys
import time
import logging
import traceback
import uuid
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Depends, File, UploadFile, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import io

# Імпортуємо модулі для моніторингу
from app.monitoring import setup_monitoring, http_request_counter, http_request_duration
from app.model import ThreatDetectionModel
from app.schemas import PredictionResponse
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

# Налаштовуємо моніторинг
setup_monitoring(app)

# Додаємо CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Додаємо middleware для відстеження часу запитів
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

@app.get("/")
async def root():
    """
    Кореневий ендпоінт для перевірки працездатності API
    """
    http_request_counter.labels(method="GET", endpoint="/", status="200").inc()
    return {"status": "ok", "message": "Threat Detection API працює"}

@app.get("/health")
async def health_check():
    """
    Ендпоінт для перевірки здоров'я сервісу
    """
    http_request_counter.labels(method="GET", endpoint="/health", status="200").inc()
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
        http_request_counter.labels(method="POST", endpoint="/predict", status="400").inc()
        raise HTTPException(status_code=400, detail="Файл повинен бути зображенням")

    try:
        start_time = time.time()

        # Читаємо зображення
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))

        # Робимо передбачення
        model = app.state.model
        result = model.predict(image, confidence_threshold=threshold)

        # Вимірюємо час передбачення
        prediction_time = time.time() - start_time
        http_request_duration.observe(prediction_time)

        # Формуємо відповідь
        response = PredictionResponse(
            request_id=request_id,
            threats=result["threats"],
            recommendations=result["recommendations"],
            details=result["details"]
        )

        # Додаємо асинхронне завдання для логування результатів
        background_tasks.add_task(log_prediction_result, request_id, len(result["threats"]))

        # Оновлюємо лічильник запитів
        http_request_counter.labels(method="POST", endpoint="/predict", status="200").inc()

        return response

    except Exception as e:
        logger.error(f"Помилка при обробці зображення: {str(e)}")
        logger.error(traceback.format_exc())
        http_request_counter.labels(method="POST", endpoint="/predict", status="500").inc()
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
    http_request_counter.labels(method=request.method, endpoint=request.url.path, status="500").inc()
    return JSONResponse(
        status_code=500,
        content={"detail": "Виникла внутрішня помилка сервера"}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
