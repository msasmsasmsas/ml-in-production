#!/usr/bin/env python

'''
FastAPI сервер з асинхронним інференсом моделей
'''

import os
import time
import asyncio
import uuid
import json
import logging
import threading
from typing import Dict, List, Any, Optional, Union

import uvicorn
import numpy as np
from fastapi import FastAPI, File, UploadFile, BackgroundTasks, HTTPException, status, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Kafka інтеграція
from kafka_queue_service import KafkaQueueService

# Налаштування логування
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('async_server')

# Створення FastAPI додатку
app = FastAPI(
    title="Async Model Inference API",
    description="API для асинхронного інференсу моделей машинного навчання",
    version="1.0.0"
)

# Конфігурація
KAFKA_BOOTSTRAP_SERVERS = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
REQUEST_TOPIC = os.environ.get("KAFKA_REQUEST_TOPIC", "model-inference-requests")
RESPONSE_TOPIC = os.environ.get("KAFKA_RESPONSE_TOPIC", "model-inference-responses")
CONSUMER_GROUP = os.environ.get("KAFKA_CONSUMER_GROUP", "model-inference-group")
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "2"))

# Створення сервісу черги Kafka
queue_service = KafkaQueueService(
    bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
    request_topic=REQUEST_TOPIC,
    response_topic=RESPONSE_TOPIC,
    consumer_group=CONSUMER_GROUP,
    num_workers=NUM_WORKERS
)

# Словник для відстеження статусу запитів
request_status = {}
status_lock = threading.Lock()

# Моделі даних
class InferenceRequest(BaseModel):
    model_name: str
    data: Any
    webhook_url: Optional[str] = None

class InferenceResponse(BaseModel):
    request_id: str
    status: str
    result: Optional[Any] = None
    error: Optional[str] = None

class RequestStatusResponse(BaseModel):
    request_id: str
    status: str
    created_at: float
    completed_at: Optional[float] = None
    result_url: Optional[str] = None

# Обробники
def inference_callback(response):
    '''
    Callback для обробки результату інференсу

    Параметри:
    -----------
    response: відповідь від сервісу черги
    '''
    request_id = response.get('request_id')

    with status_lock:
        if request_id in request_status:
            status_info = request_status[request_id]

            if response.get('success', False):
                status_info['status'] = 'completed'
                status_info['result'] = response.get('result')
            else:
                status_info['status'] = 'failed'
                status_info['error'] = response.get('error', 'Unknown error')

            status_info['completed_at'] = time.time()

    # Відправка webhook, якщо вказано
    webhook_url = status_info.get('webhook_url')
    if webhook_url:
        try:
            import requests
            requests.post(
                webhook_url,
                json={
                    'request_id': request_id,
                    'status': status_info['status'],
                    'result': status_info.get('result'),
                    'error': status_info.get('error')
                },
                timeout=5
            )
        except Exception as e:
            logger.error(f"Помилка відправки webhook для {request_id}: {e}")

@app.on_event("startup")
async def startup_event():
    '''
    Ініціалізація при запуску сервера
    '''
    # Запуск сервісу Kafka черги
    queue_service.start()
    logger.info("Сервіс асинхронного інференсу запущено")

@app.on_event("shutdown")
async def shutdown_event():
    '''
    Зупинка при завершенні роботи сервера
    '''
    # Зупинка сервісу Kafka черги
    queue_service.stop()
    logger.info("Сервіс асинхронного інференсу зупинено")

@app.post("/inference", response_model=InferenceResponse)
async def submit_inference(request: InferenceRequest):
    '''
    Ендпоінт для асинхронного інференсу моделі
    '''
    request_id = str(uuid.uuid4())

    # Збереження інформації про запит
    with status_lock:
        request_status[request_id] = {
            'status': 'submitted',
            'created_at': time.time(),
            'webhook_url': request.webhook_url
        }

    try:
        # Відправка запиту в чергу
        queue_service.submit_inference_request(
            model_name=request.model_name,
            data=request.data,
            callback=inference_callback
        )

        return InferenceResponse(
            request_id=request_id,
            status='submitted'
        )
    except Exception as e:
        logger.error(f"Помилка відправки запиту: {e}")
        with status_lock:
            if request_id in request_status:
                request_status[request_id]['status'] = 'failed'
                request_status[request_id]['error'] = str(e)
                request_status[request_id]['completed_at'] = time.time()

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Помилка відправки запиту: {str(e)}"
        )

@app.post("/inference/file", response_model=InferenceResponse)
async def submit_file_inference(
    model_name: str = Query(..., description="Назва моделі для інференсу"),
    file: UploadFile = File(..., description="Файл для інференсу"),
    webhook_url: Optional[str] = Query(None, description="URL для webhook сповіщення")
):
    '''
    Ендпоінт для асинхронного інференсу моделі з файлом
    '''
    request_id = str(uuid.uuid4())

    # Збереження інформації про запит
    with status_lock:
        request_status[request_id] = {
            'status': 'submitted',
            'created_at': time.time(),
            'webhook_url': webhook_url
        }

    try:
        # Читання файлу
        contents = await file.read()

        # Відправка запиту в чергу
        queue_service.submit_inference_request(
            model_name=model_name,
            data={
                'filename': file.filename,
                'content_type': file.content_type,
                'data': contents.hex()  # Конвертація байтів у hex для безпечної серіалізації JSON
            },
            callback=inference_callback
        )

        return InferenceResponse(
            request_id=request_id,
            status='submitted'
        )
    except Exception as e:
        logger.error(f"Помилка відправки файлового запиту: {e}")
        with status_lock:
            if request_id in request_status:
                request_status[request_id]['status'] = 'failed'
                request_status[request_id]['error'] = str(e)
                request_status[request_id]['completed_at'] = time.time()

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Помилка відправки файлового запиту: {str(e)}"
        )

@app.get("/status/{request_id}", response_model=RequestStatusResponse)
async def get_request_status(request_id: str):
    '''
    Отримання статусу запиту за ідентифікатором
    '''
    with status_lock:
        if request_id not in request_status:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Запит з ідентифікатором {request_id} не знайдено"
            )

        status_info = request_status[request_id].copy()

    return RequestStatusResponse(
        request_id=request_id,
        status=status_info['status'],
        created_at=status_info['created_at'],
        completed_at=status_info.get('completed_at'),
        result_url=f"/result/{request_id}" if status_info.get('status') == 'completed' else None
    )

@app.get("/result/{request_id}", response_model=InferenceResponse)
async def get_result(request_id: str):
    '''
    Отримання результату запиту за ідентифікатором
    '''
    with status_lock:
        if request_id not in request_status:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Запит з ідентифікатором {request_id} не знайдено"
            )

        status_info = request_status[request_id]

        if status_info['status'] == 'submitted':
            return InferenceResponse(
                request_id=request_id,
                status='submitted'
            )
        elif status_info['status'] == 'failed':
            return InferenceResponse(
                request_id=request_id,
                status='failed',
                error=status_info.get('error', 'Unknown error')
            )
        elif status_info['status'] == 'completed':
            return InferenceResponse(
                request_id=request_id,
                status='completed',
                result=status_info.get('result')
            )

@app.delete("/status/{request_id}")
async def delete_request_status(request_id: str):
    '''
    Видалення інформації про запит
    '''
    with status_lock:
        if request_id not in request_status:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Запит з ідентифікатором {request_id} не знайдено"
            )

        del request_status[request_id]

    return JSONResponse(content={"message": f"Інформацію про запит {request_id} видалено"})

@app.get("/health")
async def health_check():
    '''
    Ендпоінт перевірки стану сервера
    '''
    return {"status": "ok"}

if __name__ == "__main__":
    # Запуск сервера
    uvicorn.run(
        "async_fastapi_server:app",
        host="0.0.0.0",
        port=8000,
        reload=False
    )
