# новлена версія для PR
# новлена версія для PR
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
РЎС…РµРјРё РґР°РЅРёС… РґР»СЏ API СЂРѕР·РїС–Р·РЅР°РІР°РЅРЅСЏ Р·Р°РіСЂРѕР·
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class ThreatPrediction(BaseModel):
    """
    РњРѕРґРµР»СЊ РґР»СЏ РїРµСЂРµРґР±Р°С‡РµРЅРЅСЏ РѕРєСЂРµРјРѕС— Р·Р°РіСЂРѕР·Рё
    """
    type: str = Field(..., description="РўРёРї Р·Р°РіСЂРѕР·Рё (С…РІРѕСЂРѕР±Р°, С€РєС–РґРЅРёРє, Р±СѓСЂ'СЏРЅ)")
    name: str = Field(..., description="РќР°Р·РІР° Р·Р°РіСЂРѕР·Рё")
    confidence: float = Field(..., description="Р’РїРµРІРЅРµРЅС–СЃС‚СЊ РїРµСЂРµРґР±Р°С‡РµРЅРЅСЏ (0-1)")

class PredictionDetails(BaseModel):
    """
    Р”РѕРґР°С‚РєРѕРІС– РґРµС‚Р°Р»С– РїРµСЂРµРґР±Р°С‡РµРЅРЅСЏ
    """
    severity: Optional[str] = Field(None, description="Р С–РІРµРЅСЊ СЃРµСЂР№РѕР·РЅРѕСЃС‚С– (РЅРёР·СЊРєР°, СЃРµСЂРµРґРЅСЏ, РІРёСЃРѕРєР°)")
    processing_time_ms: Optional[int] = Field(None, description="Р§Р°СЃ РѕР±СЂРѕР±РєРё РІ РјС–Р»С–СЃРµРєСѓРЅРґР°С…")
    image_quality: Optional[str] = Field(None, description="РћС†С–РЅРєР° СЏРєРѕСЃС‚С– Р·РѕР±СЂР°Р¶РµРЅРЅСЏ")
    model_version: Optional[str] = Field(None, description="Р’РµСЂСЃС–СЏ РјРѕРґРµР»С–")

class PredictionResponse(BaseModel):
    """
    Р’С–РґРїРѕРІС–РґСЊ Р· СЂРµР·СѓР»СЊС‚Р°С‚Р°РјРё РїРµСЂРµРґР±Р°С‡РµРЅРЅСЏ
    """
    request_id: str = Field(..., description="РЈРЅС–РєР°Р»СЊРЅРёР№ С–РґРµРЅС‚РёС„С–РєР°С‚РѕСЂ Р·Р°РїРёС‚Сѓ")
    threats: List[ThreatPrediction] = Field([], description="РЎРїРёСЃРѕРє РІРёСЏРІР»РµРЅРёС… Р·Р°РіСЂРѕР·")
    recommendations: List[str] = Field([], description="Р РµРєРѕРјРµРЅРґР°С†С–С— РЅР° РѕСЃРЅРѕРІС– РІРёСЏРІР»РµРЅРёС… Р·Р°РіСЂРѕР·")
    details: Optional[Dict[str, Any]] = Field(None, description="Р”РѕРґР°С‚РєРѕРІС– РґРµС‚Р°Р»С– РїРµСЂРµРґР±Р°С‡РµРЅРЅСЏ")


