# новлена версія для PR
# новлена версія для PR
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
РњРѕРґСѓР»СЊ РґР»СЏ РЅР°Р»Р°С€С‚СѓРІР°РЅРЅСЏ РјРѕРЅС–С‚РѕСЂРёРЅРіСѓ Р· Prometheus
"""

import time
import logging
import psutil
from prometheus_client import Counter, Histogram, Gauge, Summary, Info, make_asgi_app
from prometheus_client import multiprocess, CollectorRegistry
from fastapi import FastAPI

from app.config import settings

logger = logging.getLogger(__name__)

# Р’РёР·РЅР°С‡Р°С”РјРѕ РјРµС‚СЂРёРєРё
http_request_counter = Counter(
    'http_requests_total', 
    'РљС–Р»СЊРєС–СЃС‚СЊ HTTP Р·Р°РїРёС‚С–РІ',
    ['method', 'endpoint', 'status']
)

http_request_duration = Histogram(
    'http_request_duration_seconds',
    'РўСЂРёРІР°Р»С–СЃС‚СЊ HTTP Р·Р°РїРёС‚С–РІ',
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
)

prediction_count = Counter(
    'prediction_count_total',
    'РљС–Р»СЊРєС–СЃС‚СЊ РїРµСЂРµРґР±Р°С‡РµРЅСЊ',
    ['result']
)

prediction_processing_time = Summary(
    'prediction_processing_time_seconds',
    'Р§Р°СЃ РѕР±СЂРѕР±РєРё РїРµСЂРµРґР±Р°С‡РµРЅРЅСЏ'
)

prediction_confidence = Histogram(
    'prediction_confidence',
    'Р РѕР·РїРѕРґС–Р» СЂС–РІРЅС–РІ РІРїРµРІРЅРµРЅРѕСЃС‚С– РїРµСЂРµРґР±Р°С‡РµРЅСЊ',
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

prediction_drift_score = Gauge(
    'prediction_drift_score',
    'РћС†С–РЅРєР° РґСЂРµР№С„Сѓ СЂРѕР·РїРѕРґС–Р»Сѓ РїРµСЂРµРґР±Р°С‡РµРЅСЊ'
)

system_memory_usage = Gauge(
    'system_memory_usage_bytes',
    'Р’РёРєРѕСЂРёСЃС‚Р°РЅРЅСЏ РїР°Рј\'СЏС‚С– СЃРёСЃС‚РµРјРѕСЋ'
)

system_cpu_usage = Gauge(
    'system_cpu_usage_percent',
    'Р’РёРєРѕСЂРёСЃС‚Р°РЅРЅСЏ CPU СЃРёСЃС‚РµРјРѕСЋ'
)

model_info = Info(
    'model_info',
    'Р†РЅС„РѕСЂРјР°С†С–СЏ РїСЂРѕ РјРѕРґРµР»СЊ'
)

def collect_system_metrics():
    """
    Р—Р±РёСЂР°С” СЃРёСЃС‚РµРјРЅС– РјРµС‚СЂРёРєРё (РїР°Рј'СЏС‚СЊ, CPU)
    """
    system_memory_usage.set(psutil.virtual_memory().used)
    system_cpu_usage.set(psutil.cpu_percent())

def setup_monitoring(app: FastAPI):
    """
    РќР°Р»Р°С€С‚СѓРІР°РЅРЅСЏ РјРѕРЅС–С‚РѕСЂРёРЅРіСѓ РґР»СЏ FastAPI РґРѕРґР°С‚РєСѓ

    Args:
        app: Р•РєР·РµРјРїР»СЏСЂ FastAPI РґРѕРґР°С‚РєСѓ
    """
    logger.info("РќР°Р»Р°С€С‚СѓРІР°РЅРЅСЏ РјРѕРЅС–С‚РѕСЂРёРЅРіСѓ Prometheus")

    # Р’СЃС‚Р°РЅРѕРІР»СЋС”РјРѕ С–РЅС„РѕСЂРјР°С†С–СЋ РїСЂРѕ РјРѕРґРµР»СЊ
    model_info.info({
        'name': 'threat_detection_model',
        'version': '1.0.0',
        'framework': 'pytorch',
        'description': 'РњРѕРґРµР»СЊ РґР»СЏ РІРёСЏРІР»РµРЅРЅСЏ Р·Р°РіСЂРѕР· СЃС–Р»СЊСЃСЊРєРѕРіРѕСЃРїРѕРґР°СЂСЃСЊРєРёРј РєСѓР»СЊС‚СѓСЂР°Рј'
    })

    # Р’СЃС‚Р°РЅРѕРІР»СЋС”РјРѕ Р·РЅР°С‡РµРЅРЅСЏ РґСЂРµР№С„Сѓ Р·Р° Р·Р°РјРѕРІС‡СѓРІР°РЅРЅСЏРј
    prediction_drift_score.set(0.0)

    # Р”РѕРґР°С”РјРѕ РµРЅРґРїРѕС–РЅС‚ /metrics РґР»СЏ Prometheus
    metrics_app = make_asgi_app()
    app.mount("/metrics", metrics_app)

    # Р—Р°РїСѓСЃРєР°С”РјРѕ СЂРµРіСѓР»СЏСЂРЅРµ РѕРЅРѕРІР»РµРЅРЅСЏ СЃРёСЃС‚РµРјРЅРёС… РјРµС‚СЂРёРє
    @app.on_event("startup")
    async def startup_monitoring_event():
        import threading
        import asyncio

        def collect_metrics_periodically():
            while True:
                collect_system_metrics()
                time.sleep(15)  # РћРЅРѕРІР»СЋС”РјРѕ РєРѕР¶РЅС– 15 СЃРµРєСѓРЅРґ

        # Р—Р°РїСѓСЃРєР°С”РјРѕ Р·Р±С–СЂ РјРµС‚СЂРёРє Сѓ РѕРєСЂРµРјРѕРјСѓ РїРѕС‚РѕС†С–
        metrics_thread = threading.Thread(target=collect_metrics_periodically, daemon=True)
        metrics_thread.start()

        logger.info("РњРѕРЅС–С‚РѕСЂРёРЅРі СЃРёСЃС‚РµРјРЅРёС… РјРµС‚СЂРёРє Р·Р°РїСѓС‰РµРЅРѕ")

    logger.info("РњРѕРЅС–С‚РѕСЂРёРЅРі Prometheus СѓСЃРїС–С€РЅРѕ РЅР°Р»Р°С€С‚РѕРІР°РЅРѕ")


