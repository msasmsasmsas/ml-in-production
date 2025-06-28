#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Модуль для налаштування моніторингу з Prometheus
"""

import time
import logging
import psutil
from prometheus_client import Counter, Histogram, Gauge, Summary, Info, make_asgi_app
from prometheus_client import multiprocess, CollectorRegistry
from fastapi import FastAPI

from app.config import settings

logger = logging.getLogger(__name__)

# Визначаємо метрики
http_request_counter = Counter(
    'http_requests_total', 
    'Кількість HTTP запитів',
    ['method', 'endpoint', 'status']
)

http_request_duration = Histogram(
    'http_request_duration_seconds',
    'Тривалість HTTP запитів',
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
)

prediction_count = Counter(
    'prediction_count_total',
    'Кількість передбачень',
    ['result']
)

prediction_processing_time = Summary(
    'prediction_processing_time_seconds',
    'Час обробки передбачення'
)

prediction_confidence = Histogram(
    'prediction_confidence',
    'Розподіл рівнів впевненості передбачень',
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

prediction_drift_score = Gauge(
    'prediction_drift_score',
    'Оцінка дрейфу розподілу передбачень'
)

system_memory_usage = Gauge(
    'system_memory_usage_bytes',
    'Використання пам\'яті системою'
)

system_cpu_usage = Gauge(
    'system_cpu_usage_percent',
    'Використання CPU системою'
)

model_info = Info(
    'model_info',
    'Інформація про модель'
)

def collect_system_metrics():
    """
    Збирає системні метрики (пам'ять, CPU)
    """
    system_memory_usage.set(psutil.virtual_memory().used)
    system_cpu_usage.set(psutil.cpu_percent())

def setup_monitoring(app: FastAPI):
    """
    Налаштування моніторингу для FastAPI додатку

    Args:
        app: Екземпляр FastAPI додатку
    """
    logger.info("Налаштування моніторингу Prometheus")

    # Встановлюємо інформацію про модель
    model_info.info({
        'name': 'threat_detection_model',
        'version': '1.0.0',
        'framework': 'pytorch',
        'description': 'Модель для виявлення загроз сільськогосподарським культурам'
    })

    # Встановлюємо значення дрейфу за замовчуванням
    prediction_drift_score.set(0.0)

    # Додаємо ендпоінт /metrics для Prometheus
    metrics_app = make_asgi_app()
    app.mount("/metrics", metrics_app)

    # Запускаємо регулярне оновлення системних метрик
    @app.on_event("startup")
    async def startup_monitoring_event():
        import threading
        import asyncio

        def collect_metrics_periodically():
            while True:
                collect_system_metrics()
                time.sleep(15)  # Оновлюємо кожні 15 секунд

        # Запускаємо збір метрик у окремому потоці
        metrics_thread = threading.Thread(target=collect_metrics_periodically, daemon=True)
        metrics_thread.start()

        logger.info("Моніторинг системних метрик запущено")

    logger.info("Моніторинг Prometheus успішно налаштовано")
