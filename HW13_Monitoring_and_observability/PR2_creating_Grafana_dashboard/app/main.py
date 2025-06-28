# новлена версія для PR
# новлена версія для PR
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Р“РѕР»РѕРІРЅРёР№ С„Р°Р№Р» РґРѕРґР°С‚РєСѓ Р· С–РЅС‚РµРіСЂР°С†С–С”СЋ Prometheus РґР»СЏ РјРѕРЅС–С‚РѕСЂРёРЅРіСѓ
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

# Р†РјРїРѕСЂС‚СѓС”РјРѕ РјРѕРґСѓР»С– РґР»СЏ РјРѕРЅС–С‚РѕСЂРёРЅРіСѓ
from app.monitoring import setup_monitoring, http_request_counter, http_request_duration
from app.model import ThreatDetectionModel
from app.schemas import PredictionResponse
from app.config import settings

# РќР°Р»Р°С€С‚СѓРІР°РЅРЅСЏ Р»РѕРіСѓРІР°РЅРЅСЏ
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# РЎС‚РІРѕСЂСЋС”РјРѕ РєРѕРЅС‚РµРєСЃС‚РЅРёР№ РјРµРЅРµРґР¶РµСЂ РґР»СЏ Р·Р°РїСѓСЃРєСѓ С‚Р° Р·СѓРїРёРЅРєРё СЃРµСЂРІС–СЃС–РІ
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Р’РёРєРѕРЅСѓС”С‚СЊСЃСЏ РїСЂРё СЃС‚Р°СЂС‚С–
    logger.info("Р†РЅС–С†С–Р°Р»С–Р·Р°С†С–СЏ РґРѕРґР°С‚РєСѓ...")
    # Р†РЅС–С†С–Р°Р»С–Р·СѓС”РјРѕ РјРѕРґРµР»СЊ РїСЂРё СЃС‚Р°СЂС‚С–
    app.state.model = ThreatDetectionModel()
    yield
    # Р’РёРєРѕРЅСѓС”С‚СЊСЃСЏ РїСЂРё Р·СѓРїРёРЅС†С–
    logger.info("Р—Р°РІРµСЂС€РµРЅРЅСЏ СЂРѕР±РѕС‚Рё РґРѕРґР°С‚РєСѓ...")

# РЎС‚РІРѕСЂСЋС”РјРѕ FastAPI РґРѕРґР°С‚РѕРє
app = FastAPI(
    title="Threat Detection API",
    description="API РґР»СЏ РІРёСЏРІР»РµРЅРЅСЏ Р·Р°РіСЂРѕР· СЃС–Р»СЊСЃСЊРєРѕРіРѕСЃРїРѕРґР°СЂСЃСЊРєРёРј РєСѓР»СЊС‚СѓСЂР°Рј",
    version="1.0.0",
    lifespan=lifespan
)

# РќР°Р»Р°С€С‚РѕРІСѓС”РјРѕ РјРѕРЅС–С‚РѕСЂРёРЅРі
setup_monitoring(app)

# Р”РѕРґР°С”РјРѕ CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Р”РѕРґР°С”РјРѕ middleware РґР»СЏ РІС–РґСЃС‚РµР¶РµРЅРЅСЏ С‡Р°СЃСѓ Р·Р°РїРёС‚С–РІ
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
    РљРѕСЂРµРЅРµРІРёР№ РµРЅРґРїРѕС–РЅС‚ РґР»СЏ РїРµСЂРµРІС–СЂРєРё РїСЂР°С†РµР·РґР°С‚РЅРѕСЃС‚С– API
    """
    http_request_counter.labels(method="GET", endpoint="/", status="200").inc()
    return {"status": "ok", "message": "Threat Detection API РїСЂР°С†СЋС”"}

@app.get("/health")
async def health_check():
    """
    Р•РЅРґРїРѕС–РЅС‚ РґР»СЏ РїРµСЂРµРІС–СЂРєРё Р·РґРѕСЂРѕРІ'СЏ СЃРµСЂРІС–СЃСѓ
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
    Р•РЅРґРїРѕС–РЅС‚ РґР»СЏ РїРµСЂРµРґР±Р°С‡РµРЅРЅСЏ Р·Р°РіСЂРѕР· РЅР° Р·РѕР±СЂР°Р¶РµРЅРЅС–

    - **file**: Р·РѕР±СЂР°Р¶РµРЅРЅСЏ РґР»СЏ Р°РЅР°Р»С–Р·Сѓ
    - **threshold**: РїРѕСЂС–Рі РІРїРµРІРЅРµРЅРѕСЃС‚С– (0.0-1.0)
    """
    request_id = str(uuid.uuid4())
    logger.info(f"РћС‚СЂРёРјР°РЅРѕ Р·Р°РїРёС‚ РЅР° РїРµСЂРµРґР±Р°С‡РµРЅРЅСЏ [ID: {request_id}]")

    # РџРµСЂРµРІС–СЂСЏС”РјРѕ, С‡Рё С†Рµ Р·РѕР±СЂР°Р¶РµРЅРЅСЏ
    if not file.content_type.startswith("image/"):
        http_request_counter.labels(method="POST", endpoint="/predict", status="400").inc()
        raise HTTPException(status_code=400, detail="Р¤Р°Р№Р» РїРѕРІРёРЅРµРЅ Р±СѓС‚Рё Р·РѕР±СЂР°Р¶РµРЅРЅСЏРј")

    try:
        start_time = time.time()

        # Р§РёС‚Р°С”РјРѕ Р·РѕР±СЂР°Р¶РµРЅРЅСЏ
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))

        # Р РѕР±РёРјРѕ РїРµСЂРµРґР±Р°С‡РµРЅРЅСЏ
        model = app.state.model
        result = model.predict(image, confidence_threshold=threshold)

        # Р’РёРјС–СЂСЋС”РјРѕ С‡Р°СЃ РїРµСЂРµРґР±Р°С‡РµРЅРЅСЏ
        prediction_time = time.time() - start_time
        http_request_duration.observe(prediction_time)

        # Р¤РѕСЂРјСѓС”РјРѕ РІС–РґРїРѕРІС–РґСЊ
        response = PredictionResponse(
            request_id=request_id,
            threats=result["threats"],
            recommendations=result["recommendations"],
            details=result["details"]
        )

        # Р”РѕРґР°С”РјРѕ Р°СЃРёРЅС…СЂРѕРЅРЅРµ Р·Р°РІРґР°РЅРЅСЏ РґР»СЏ Р»РѕРіСѓРІР°РЅРЅСЏ СЂРµР·СѓР»СЊС‚Р°С‚С–РІ
        background_tasks.add_task(log_prediction_result, request_id, len(result["threats"]))

        # РћРЅРѕРІР»СЋС”РјРѕ Р»С–С‡РёР»СЊРЅРёРє Р·Р°РїРёС‚С–РІ
        http_request_counter.labels(method="POST", endpoint="/predict", status="200").inc()

        return response

    except Exception as e:
        logger.error(f"РџРѕРјРёР»РєР° РїСЂРё РѕР±СЂРѕР±С†С– Р·РѕР±СЂР°Р¶РµРЅРЅСЏ: {str(e)}")
        logger.error(traceback.format_exc())
        http_request_counter.labels(method="POST", endpoint="/predict", status="500").inc()
        raise HTTPException(status_code=500, detail=f"РџРѕРјРёР»РєР° РѕР±СЂРѕР±РєРё: {str(e)}")

def log_prediction_result(request_id: str, threat_count: int):
    """
    Р¤СѓРЅРєС†С–СЏ РґР»СЏ Р°СЃРёРЅС…СЂРѕРЅРЅРѕРіРѕ Р»РѕРіСѓРІР°РЅРЅСЏ СЂРµР·СѓР»СЊС‚Р°С‚С–РІ РїРµСЂРµРґР±Р°С‡РµРЅРЅСЏ
    """
    logger.info(f"Р—Р°РІРµСЂС€РµРЅРѕ РїРµСЂРµРґР±Р°С‡РµРЅРЅСЏ [ID: {request_id}], РІРёСЏРІР»РµРЅРѕ Р·Р°РіСЂРѕР·: {threat_count}")

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """
    Р“Р»РѕР±Р°Р»СЊРЅРёР№ РѕР±СЂРѕР±РЅРёРє РІРёРЅСЏС‚РєС–РІ
    """
    logger.error(f"РќРµРїРµСЂРµРґР±Р°С‡РµРЅР° РїРѕРјРёР»РєР°: {str(exc)}")
    logger.error(traceback.format_exc())
    http_request_counter.labels(method=request.method, endpoint=request.url.path, status="500").inc()
    return JSONResponse(
        status_code=500,
        content={"detail": "Р’РёРЅРёРєР»Р° РІРЅСѓС‚СЂС–С€РЅСЏ РїРѕРјРёР»РєР° СЃРµСЂРІРµСЂР°"}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)


