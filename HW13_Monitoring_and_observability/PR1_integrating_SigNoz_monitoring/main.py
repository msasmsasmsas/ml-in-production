# новлена версія для PR
# новлена версія для PR
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Р“РѕР»РѕРІРЅРёР№ С„Р°Р№Р» РґРѕРґР°С‚РєСѓ Р· С–РЅС‚РµРіСЂР°С†С–С”СЋ РјРѕРЅС–С‚РѕСЂРёРЅРіСѓ SigNoz С‡РµСЂРµР· OpenTelemetry
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

# Р†РјРїРѕСЂС‚СѓС”РјРѕ РјРѕРґСѓР»С– РґР»СЏ SigNoz/OpenTelemetry
from app.telemetry import setup_telemetry
from app.model import ThreatDetectionModel
from app.schemas import PredictionResponse, PredictionRequest
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

# Р†РЅС–С†С–Р°Р»С–Р·СѓС”РјРѕ С‚РµР»РµРјРµС‚СЂС–СЋ
setup_telemetry(service_name="threat-detection-api")

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

# Р”РѕРґР°С”РјРѕ CORS middleware
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
    РљРѕСЂРµРЅРµРІРёР№ РµРЅРґРїРѕС–РЅС‚ РґР»СЏ РїРµСЂРµРІС–СЂРєРё РїСЂР°С†РµР·РґР°С‚РЅРѕСЃС‚С– API
    """
    return {"status": "ok", "message": "Threat Detection API РїСЂР°С†СЋС”"}

@app.get("/health")
async def health_check():
    """
    Р•РЅРґРїРѕС–РЅС‚ РґР»СЏ РїРµСЂРµРІС–СЂРєРё Р·РґРѕСЂРѕРІ'СЏ СЃРµСЂРІС–СЃСѓ
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
    Р•РЅРґРїРѕС–РЅС‚ РґР»СЏ РїРµСЂРµРґР±Р°С‡РµРЅРЅСЏ Р·Р°РіСЂРѕР· РЅР° Р·РѕР±СЂР°Р¶РµРЅРЅС–

    - **file**: Р·РѕР±СЂР°Р¶РµРЅРЅСЏ РґР»СЏ Р°РЅР°Р»С–Р·Сѓ
    - **threshold**: РїРѕСЂС–Рі РІРїРµРІРЅРµРЅРѕСЃС‚С– (0.0-1.0)
    """
    request_id = str(uuid.uuid4())
    logger.info(f"РћС‚СЂРёРјР°РЅРѕ Р·Р°РїРёС‚ РЅР° РїРµСЂРµРґР±Р°С‡РµРЅРЅСЏ [ID: {request_id}]")

    # РџРµСЂРµРІС–СЂСЏС”РјРѕ, С‡Рё С†Рµ Р·РѕР±СЂР°Р¶РµРЅРЅСЏ
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Р¤Р°Р№Р» РїРѕРІРёРЅРµРЅ Р±СѓС‚Рё Р·РѕР±СЂР°Р¶РµРЅРЅСЏРј")

    try:
        # Р§РёС‚Р°С”РјРѕ Р·РѕР±СЂР°Р¶РµРЅРЅСЏ
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))

        # Р РѕР±РёРјРѕ РїРµСЂРµРґР±Р°С‡РµРЅРЅСЏ
        model = app.state.model
        result = model.predict(image, confidence_threshold=threshold)

        # Р¤РѕСЂРјСѓС”РјРѕ РІС–РґРїРѕРІС–РґСЊ
        response = PredictionResponse(
            request_id=request_id,
            threats=result["threats"],
            recommendations=result["recommendations"],
            details=result["details"]
        )

        # Р”РѕРґР°С”РјРѕ Р°СЃРёРЅС…СЂРѕРЅРЅРµ Р·Р°РІРґР°РЅРЅСЏ РґР»СЏ Р»РѕРіСѓРІР°РЅРЅСЏ СЂРµР·СѓР»СЊС‚Р°С‚С–РІ
        background_tasks.add_task(log_prediction_result, request_id, len(result["threats"]))

        return response

    except Exception as e:
        logger.error(f"РџРѕРјРёР»РєР° РїСЂРё РѕР±СЂРѕР±С†С– Р·РѕР±СЂР°Р¶РµРЅРЅСЏ: {str(e)}")
        logger.error(traceback.format_exc())
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
    return JSONResponse(
        status_code=500,
        content={"detail": "Р’РёРЅРёРєР»Р° РІРЅСѓС‚СЂС–С€РЅСЏ РїРѕРјРёР»РєР° СЃРµСЂРІРµСЂР°"}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)


