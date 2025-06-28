#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Схеми даних для API розпізнавання загроз
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class ThreatPrediction(BaseModel):
    """
    Модель для передбачення окремої загрози
    """
    type: str = Field(..., description="Тип загрози (хвороба, шкідник, бур'ян)")
    name: str = Field(..., description="Назва загрози")
    confidence: float = Field(..., description="Впевненість передбачення (0-1)")

class PredictionDetails(BaseModel):
    """
    Додаткові деталі передбачення
    """
    severity: Optional[str] = Field(None, description="Рівень серйозності (низька, середня, висока)")
    processing_time_ms: Optional[int] = Field(None, description="Час обробки в мілісекундах")
    image_quality: Optional[str] = Field(None, description="Оцінка якості зображення")
    model_version: Optional[str] = Field(None, description="Версія моделі")

class PredictionResponse(BaseModel):
    """
    Відповідь з результатами передбачення
    """
    request_id: str = Field(..., description="Унікальний ідентифікатор запиту")
    threats: List[ThreatPrediction] = Field([], description="Список виявлених загроз")
    recommendations: List[str] = Field([], description="Рекомендації на основі виявлених загроз")
    details: Optional[Dict[str, Any]] = Field(None, description="Додаткові деталі передбачення")
