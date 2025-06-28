# новлена версія для PR
#!/usr/bin/env python

'''
FastAPI СЃРµСЂРІРµСЂ Р· Р°СЃРёРЅС…СЂРѕРЅРЅРёРј С–РЅС„РµСЂРµРЅСЃРѕРј РјРѕРґРµР»РµР№
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

# Kafka С–РЅС‚РµРіСЂР°С†С–СЏ
from kafka_queue_service import KafkaQueueService

# РќР°Р»Р°С€С‚СѓРІР°РЅРЅСЏ Р»РѕРіСѓРІР°РЅРЅСЏ
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('async_server')

# РЎС‚РІРѕСЂРµРЅРЅСЏ FastAPI РґРѕРґР°С‚РєСѓ
app = FastAPI(
    title="Async Model Inference API",
    description="API РґР»СЏ Р°СЃРёРЅС…СЂРѕРЅРЅРѕРіРѕ С–РЅС„РµСЂРµРЅСЃСѓ РјРѕРґРµР»РµР№ РјР°С€РёРЅРЅРѕРіРѕ РЅР°РІС‡Р°РЅРЅСЏ",
    version="1.0.0"
)

# РљРѕРЅС„С–РіСѓСЂР°С†С–СЏ
KAFKA_BOOTSTRAP_SERVERS = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
REQUEST_TOPIC = os.environ.get("KAFKA_REQUEST_TOPIC", "model-inference-requests")
RESPONSE_TOPIC = os.environ.get("KAFKA_RESPONSE_TOPIC", "model-inference-responses")
CONSUMER_GROUP = os.environ.get("KAFKA_CONSUMER_GROUP", "model-inference-group")
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "2"))

# РЎС‚РІРѕСЂРµРЅРЅСЏ СЃРµСЂРІС–СЃСѓ С‡РµСЂРіРё Kafka
queue_service = KafkaQueueService(
    bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
    request_topic=REQUEST_TOPIC,
    response_topic=RESPONSE_TOPIC,
    consumer_group=CONSUMER_GROUP,
    num_workers=NUM_WORKERS
)

# РЎР»РѕРІРЅРёРє РґР»СЏ РІС–РґСЃС‚РµР¶РµРЅРЅСЏ СЃС‚Р°С‚СѓСЃСѓ Р·Р°РїРёС‚С–РІ
request_status = {}
status_lock = threading.Lock()

# РњРѕРґРµР»С– РґР°РЅРёС…
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

# РћР±СЂРѕР±РЅРёРєРё
def inference_callback(response):
    '''
    Callback РґР»СЏ РѕР±СЂРѕР±РєРё СЂРµР·СѓР»СЊС‚Р°С‚Сѓ С–РЅС„РµСЂРµРЅСЃСѓ

    РџР°СЂР°РјРµС‚СЂРё:
    -----------
    response: РІС–РґРїРѕРІС–РґСЊ РІС–Рґ СЃРµСЂРІС–СЃСѓ С‡РµСЂРіРё
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

    # Р’С–РґРїСЂР°РІРєР° webhook, СЏРєС‰Рѕ РІРєР°Р·Р°РЅРѕ
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
            logger.error(f"РџРѕРјРёР»РєР° РІС–РґРїСЂР°РІРєРё webhook РґР»СЏ {request_id}: {e}")

@app.on_event("startup")
async def startup_event():
    '''
    Р†РЅС–С†С–Р°Р»С–Р·Р°С†С–СЏ РїСЂРё Р·Р°РїСѓСЃРєСѓ СЃРµСЂРІРµСЂР°
    '''
    # Р—Р°РїСѓСЃРє СЃРµСЂРІС–СЃСѓ Kafka С‡РµСЂРіРё
    queue_service.start()
    logger.info("РЎРµСЂРІС–СЃ Р°СЃРёРЅС…СЂРѕРЅРЅРѕРіРѕ С–РЅС„РµСЂРµРЅСЃСѓ Р·Р°РїСѓС‰РµРЅРѕ")

@app.on_event("shutdown")
async def shutdown_event():
    '''
    Р—СѓРїРёРЅРєР° РїСЂРё Р·Р°РІРµСЂС€РµРЅРЅС– СЂРѕР±РѕС‚Рё СЃРµСЂРІРµСЂР°
    '''
    # Р—СѓРїРёРЅРєР° СЃРµСЂРІС–СЃСѓ Kafka С‡РµСЂРіРё
    queue_service.stop()
    logger.info("РЎРµСЂРІС–СЃ Р°СЃРёРЅС…СЂРѕРЅРЅРѕРіРѕ С–РЅС„РµСЂРµРЅСЃСѓ Р·СѓРїРёРЅРµРЅРѕ")

@app.post("/inference", response_model=InferenceResponse)
async def submit_inference(request: InferenceRequest):
    '''
    Р•РЅРґРїРѕС–РЅС‚ РґР»СЏ Р°СЃРёРЅС…СЂРѕРЅРЅРѕРіРѕ С–РЅС„РµСЂРµРЅСЃСѓ РјРѕРґРµР»С–
    '''
    request_id = str(uuid.uuid4())

    # Р—Р±РµСЂРµР¶РµРЅРЅСЏ С–РЅС„РѕСЂРјР°С†С–С— РїСЂРѕ Р·Р°РїРёС‚
    with status_lock:
        request_status[request_id] = {
            'status': 'submitted',
            'created_at': time.time(),
            'webhook_url': request.webhook_url
        }

    try:
        # Р’С–РґРїСЂР°РІРєР° Р·Р°РїРёС‚Сѓ РІ С‡РµСЂРіСѓ
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
        logger.error(f"РџРѕРјРёР»РєР° РІС–РґРїСЂР°РІРєРё Р·Р°РїРёС‚Сѓ: {e}")
        with status_lock:
            if request_id in request_status:
                request_status[request_id]['status'] = 'failed'
                request_status[request_id]['error'] = str(e)
                request_status[request_id]['completed_at'] = time.time()

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"РџРѕРјРёР»РєР° РІС–РґРїСЂР°РІРєРё Р·Р°РїРёС‚Сѓ: {str(e)}"
        )

@app.post("/inference/file", response_model=InferenceResponse)
async def submit_file_inference(
    model_name: str = Query(..., description="РќР°Р·РІР° РјРѕРґРµР»С– РґР»СЏ С–РЅС„РµСЂРµРЅСЃСѓ"),
    file: UploadFile = File(..., description="Р¤Р°Р№Р» РґР»СЏ С–РЅС„РµСЂРµРЅСЃСѓ"),
    webhook_url: Optional[str] = Query(None, description="URL РґР»СЏ webhook СЃРїРѕРІС–С‰РµРЅРЅСЏ")
):
    '''
    Р•РЅРґРїРѕС–РЅС‚ РґР»СЏ Р°СЃРёРЅС…СЂРѕРЅРЅРѕРіРѕ С–РЅС„РµСЂРµРЅСЃСѓ РјРѕРґРµР»С– Р· С„Р°Р№Р»РѕРј
    '''
    request_id = str(uuid.uuid4())

    # Р—Р±РµСЂРµР¶РµРЅРЅСЏ С–РЅС„РѕСЂРјР°С†С–С— РїСЂРѕ Р·Р°РїРёС‚
    with status_lock:
        request_status[request_id] = {
            'status': 'submitted',
            'created_at': time.time(),
            'webhook_url': webhook_url
        }

    try:
        # Р§РёС‚Р°РЅРЅСЏ С„Р°Р№Р»Сѓ
        contents = await file.read()

        # Р’С–РґРїСЂР°РІРєР° Р·Р°РїРёС‚Сѓ РІ С‡РµСЂРіСѓ
        queue_service.submit_inference_request(
            model_name=model_name,
            data={
                'filename': file.filename,
                'content_type': file.content_type,
                'data': contents.hex()  # РљРѕРЅРІРµСЂС‚Р°С†С–СЏ Р±Р°Р№С‚С–РІ Сѓ hex РґР»СЏ Р±РµР·РїРµС‡РЅРѕС— СЃРµСЂС–Р°Р»С–Р·Р°С†С–С— JSON
            },
            callback=inference_callback
        )

        return InferenceResponse(
            request_id=request_id,
            status='submitted'
        )
    except Exception as e:
        logger.error(f"РџРѕРјРёР»РєР° РІС–РґРїСЂР°РІРєРё С„Р°Р№Р»РѕРІРѕРіРѕ Р·Р°РїРёС‚Сѓ: {e}")
        with status_lock:
            if request_id in request_status:
                request_status[request_id]['status'] = 'failed'
                request_status[request_id]['error'] = str(e)
                request_status[request_id]['completed_at'] = time.time()

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"РџРѕРјРёР»РєР° РІС–РґРїСЂР°РІРєРё С„Р°Р№Р»РѕРІРѕРіРѕ Р·Р°РїРёС‚Сѓ: {str(e)}"
        )

@app.get("/status/{request_id}", response_model=RequestStatusResponse)
async def get_request_status(request_id: str):
    '''
    РћС‚СЂРёРјР°РЅРЅСЏ СЃС‚Р°С‚СѓСЃСѓ Р·Р°РїРёС‚Сѓ Р·Р° С–РґРµРЅС‚РёС„С–РєР°С‚РѕСЂРѕРј
    '''
    with status_lock:
        if request_id not in request_status:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Р—Р°РїРёС‚ Р· С–РґРµРЅС‚РёС„С–РєР°С‚РѕСЂРѕРј {request_id} РЅРµ Р·РЅР°Р№РґРµРЅРѕ"
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
    РћС‚СЂРёРјР°РЅРЅСЏ СЂРµР·СѓР»СЊС‚Р°С‚Сѓ Р·Р°РїРёС‚Сѓ Р·Р° С–РґРµРЅС‚РёС„С–РєР°С‚РѕСЂРѕРј
    '''
    with status_lock:
        if request_id not in request_status:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Р—Р°РїРёС‚ Р· С–РґРµРЅС‚РёС„С–РєР°С‚РѕСЂРѕРј {request_id} РЅРµ Р·РЅР°Р№РґРµРЅРѕ"
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
    Р’РёРґР°Р»РµРЅРЅСЏ С–РЅС„РѕСЂРјР°С†С–С— РїСЂРѕ Р·Р°РїРёС‚
    '''
    with status_lock:
        if request_id not in request_status:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Р—Р°РїРёС‚ Р· С–РґРµРЅС‚РёС„С–РєР°С‚РѕСЂРѕРј {request_id} РЅРµ Р·РЅР°Р№РґРµРЅРѕ"
            )

        del request_status[request_id]

    return JSONResponse(content={"message": f"Р†РЅС„РѕСЂРјР°С†С–СЋ РїСЂРѕ Р·Р°РїРёС‚ {request_id} РІРёРґР°Р»РµРЅРѕ"})

@app.get("/health")
async def health_check():
    '''
    Р•РЅРґРїРѕС–РЅС‚ РїРµСЂРµРІС–СЂРєРё СЃС‚Р°РЅСѓ СЃРµСЂРІРµСЂР°
    '''
    return {"status": "ok"}

if __name__ == "__main__":
    # Р—Р°РїСѓСЃРє СЃРµСЂРІРµСЂР°
    uvicorn.run(
        "async_fastapi_server:app",
        host="0.0.0.0",
        port=8000,
        reload=False
    )

