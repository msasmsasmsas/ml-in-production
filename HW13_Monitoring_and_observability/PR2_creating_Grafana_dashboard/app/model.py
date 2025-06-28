# новлена версія для PR
# новлена версія для PR
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
РњРѕРґСѓР»СЊ РґР»СЏ СЂРѕР±РѕС‚Рё Р· РјРѕРґРµР»Р»СЋ РІРёСЏРІР»РµРЅРЅСЏ Р·Р°РіСЂРѕР· Р· С–РЅС‚РµРіСЂР°С†С–С”СЋ РјРµС‚СЂРёРє
"""

import os
import time
import logging
import numpy as np
from PIL import Image
import torch
from typing import Dict, Any, List

from app.monitoring import prediction_count, prediction_processing_time, prediction_confidence
from app.config import settings

logger = logging.getLogger(__name__)

class ThreatDetectionModel:
    """
    РљР»Р°СЃ РґР»СЏ СЂРѕР±РѕС‚Рё Р· РјРѕРґРµР»Р»СЋ РІРёСЏРІР»РµРЅРЅСЏ Р·Р°РіСЂРѕР· СЃС–Р»СЊСЃСЊРєРѕРіРѕСЃРїРѕРґР°СЂСЃСЊРєРёРј РєСѓР»СЊС‚СѓСЂР°Рј
    Р· С–РЅС‚РµРіСЂР°С†С–С”СЋ РјРµС‚СЂРёРє РґР»СЏ Prometheus
    """
    def __init__(self, model_path: str = None):
        """
        Р†РЅС–С†С–Р°Р»С–Р·Р°С†С–СЏ РјРѕРґРµР»С–

        Args:
            model_path (str, optional): РЁР»СЏС… РґРѕ С„Р°Р№Р»Сѓ РјРѕРґРµР»С–
        """
        self.model_path = model_path or settings.MODEL_PATH
        logger.info(f"Р†РЅС–С†С–Р°Р»С–Р·Р°С†С–СЏ РјРѕРґРµР»С– Р· {self.model_path}")

        # РЈ СЂРµР°Р»СЊРЅС–Р№ С–РјРїР»РµРјРµРЅС‚Р°С†С–С— С‚СѓС‚ Р·Р°РІР°РЅС‚Р°Р¶СѓС”РјРѕ РјРѕРґРµР»СЊ
        # self.model = torch.load(self.model_path) if os.path.exists(self.model_path) else None
        self.model = None
        self.class_names = [
            "healthy", "late_blight", "early_blight", "rust", 
            "septoria", "aphids", "thrips", "whitefly", 
            "bindweed", "nutsedge", "chickweed"
        ]
        logger.info("РњРѕРґРµР»СЊ СѓСЃРїС–С€РЅРѕ С–РЅС–С†С–Р°Р»С–Р·РѕРІР°РЅР°")

    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """
        РџРѕРїРµСЂРµРґРЅСЏ РѕР±СЂРѕР±РєР° Р·РѕР±СЂР°Р¶РµРЅРЅСЏ РґР»СЏ РјРѕРґРµР»С–

        Args:
            image (Image.Image): Р’С…С–РґРЅРµ Р·РѕР±СЂР°Р¶РµРЅРЅСЏ

        Returns:
            torch.Tensor: РџС–РґРіРѕС‚РѕРІР»РµРЅРёР№ С‚РµРЅР·РѕСЂ РґР»СЏ РјРѕРґРµР»С–
        """
        logger.debug("РџРѕРїРµСЂРµРґРЅСЏ РѕР±СЂРѕР±РєР° Р·РѕР±СЂР°Р¶РµРЅРЅСЏ")
        img_array = np.array(image.convert("RGB").resize((224, 224)))
        # РљРѕРЅРІРµСЂС‚СѓС”РјРѕ РІ С‚РµРЅР·РѕСЂ, РЅРѕСЂРјР°Р»С–Р·СѓС”РјРѕ, С‚РѕС‰Рѕ
        tensor = torch.tensor(img_array).float().permute(2, 0, 1) / 255.0
        return tensor.unsqueeze(0)

    @prediction_processing_time.time()
    def predict(self, image: Image.Image, confidence_threshold: float = 0.5) -> Dict[str, Any]:
        """
        РћР±СЂРѕР±РєР° Р·РѕР±СЂР°Р¶РµРЅРЅСЏ С‚Р° РїРѕРІРµСЂРЅРµРЅРЅСЏ РїРµСЂРµРґР±Р°С‡РµРЅСЊ

        Args:
            image (Image.Image): Р’С…С–РґРЅРµ Р·РѕР±СЂР°Р¶РµРЅРЅСЏ
            confidence_threshold (float, optional): РџРѕСЂС–Рі РІРїРµРІРЅРµРЅРѕСЃС‚С– РґР»СЏ РІРёСЏРІР»РµРЅРЅСЏ Р·Р°РіСЂРѕР·

        Returns:
            Dict[str, Any]: Р РµР·СѓР»СЊС‚Р°С‚Рё РїРµСЂРµРґР±Р°С‡РµРЅРЅСЏ
        """
        logger.info(f"Р’РёРєРѕРЅСѓС”РјРѕ РїРµСЂРµРґР±Р°С‡РµРЅРЅСЏ Р· РїРѕСЂРѕРіРѕРј РІРїРµРІРЅРµРЅРѕСЃС‚С–: {confidence_threshold}")
        start_time = time.time()

        # РџРѕРїРµСЂРµРґРЅСЏ РѕР±СЂРѕР±РєР° Р·РѕР±СЂР°Р¶РµРЅРЅСЏ
        tensor = self.preprocess_image(image)

        # РЈ СЂРµР°Р»СЊРЅС–Р№ С–РјРїР»РµРјРµРЅС‚Р°С†С–С— Р·Р°РїСѓСЃРєР°С”РјРѕ РјРѕРґРµР»СЊ
        # predictions = self.model(tensor)

        # РњРѕРєРѕРІС– РїРµСЂРµРґР±Р°С‡РµРЅРЅСЏ РґР»СЏ С‚РµСЃС‚СѓРІР°РЅРЅСЏ
        # Р РѕР±РёРјРѕ РїРµСЂРµРґР±Р°С‡РµРЅРЅСЏ РґРµС‚РµСЂРјС–РЅС–СЃС‚РёС‡РЅРёРјРё РЅР° РѕСЃРЅРѕРІС– РІРјС–СЃС‚Сѓ Р·РѕР±СЂР°Р¶РµРЅРЅСЏ
        np.random.seed(int(tensor.sum().item() * 100))

        mock_predictions = []
        # Р“РµРЅРµСЂСѓС”РјРѕ 0-3 РІРёРїР°РґРєРѕРІРёС… Р·Р°РіСЂРѕР·
        num_threats = np.random.randint(0, 4)

        for _ in range(num_threats):
            threat_type = np.random.choice(["disease", "pest", "weed"])
            confidence = np.random.uniform(0.3, 0.95)

            # Р РµС”СЃС‚СЂСѓС”РјРѕ СЂС–РІРµРЅСЊ РІРїРµРІРЅРµРЅРѕСЃС‚С–
            prediction_confidence.observe(confidence)

            # РџСЂРѕРїСѓСЃРєР°С”РјРѕ, СЏРєС‰Рѕ РЅРёР¶С‡Рµ РїРѕСЂРѕРіСѓ
            if confidence < confidence_threshold:
                continue

            # Р’РёР±РёСЂР°С”РјРѕ РЅР°Р·РІСѓ РЅР° РѕСЃРЅРѕРІС– С‚РёРїСѓ
            if threat_type == "disease":
                name = np.random.choice(["late_blight", "early_blight", "rust", "septoria"])
            elif threat_type == "pest":
                name = np.random.choice(["aphids", "thrips", "whitefly"])
            else:  # weed
                name = np.random.choice(["bindweed", "nutsedge", "chickweed"])

            mock_predictions.append({
                "type": threat_type,
                "name": name,
                "confidence": float(confidence)
            })

        # РћРЅРѕРІР»СЋС”РјРѕ Р»С–С‡РёР»СЊРЅРёРєРё РїРµСЂРµРґР±Р°С‡РµРЅСЊ
        if mock_predictions:
            prediction_count.labels(result="positive").inc()
        else:
            prediction_count.labels(result="negative").inc()

        # Р“РµРЅРµСЂСѓС”РјРѕ СЂРµРєРѕРјРµРЅРґР°С†С–С— РЅР° РѕСЃРЅРѕРІС– РІРёСЏРІР»РµРЅРёС… Р·Р°РіСЂРѕР·
        recommendations = []
        details = {}

        if not mock_predictions:
            recommendations.append("Р—Р°РіСЂРѕР· РЅРµ РІРёСЏРІР»РµРЅРѕ. РџСЂРѕРґРѕРІР¶СѓР№С‚Рµ СЂРµРіСѓР»СЏСЂРЅРёР№ РјРѕРЅС–С‚РѕСЂРёРЅРі РїРѕСЃС–РІС–РІ.")
        else:
            for pred in mock_predictions:
                if pred["type"] == "disease":
                    recommendations.append(f"Р”Р»СЏ {pred['name']}, СЂРѕР·РіР»СЏРЅСЊС‚Рµ Р·Р°СЃС‚РѕСЃСѓРІР°РЅРЅСЏ РІС–РґРїРѕРІС–РґРЅРѕРіРѕ С„СѓРЅРіС–С†РёРґСѓ.")
                elif pred["type"] == "pest":
                    recommendations.append(f"Р”Р»СЏ {pred['name']}, СЂРѕР·РіР»СЏРЅСЊС‚Рµ РІРёРєРѕСЂРёСЃС‚Р°РЅРЅСЏ С–РЅСЃРµРєС‚РёС†РёРґСѓ Р°Р±Рѕ Р±С–РѕР»РѕРіС–С‡РЅРѕРіРѕ РєРѕРЅС‚СЂРѕР»СЋ.")
                else:  # weed
                    recommendations.append(f"Р”Р»СЏ {pred['name']}, СЂРѕР·РіР»СЏРЅСЊС‚Рµ РјРµС…Р°РЅС–С‡РЅРµ РІРёРґР°Р»РµРЅРЅСЏ Р°Р±Рѕ С†С–Р»СЊРѕРІРёР№ РіРµСЂР±С–С†РёРґ.")

            # Р”РѕРґР°С”РјРѕ РґРµСЏРєС– РґРµС‚Р°Р»С–
            details = {
                "severity": "РЅРёР·СЊРєР°" if len(mock_predictions) == 1 else "СЃРµСЂРµРґРЅСЏ" if len(mock_predictions) == 2 else "РІРёСЃРѕРєР°",
                "processing_time_ms": round((time.time() - start_time) * 1000),
                "image_quality": "РґРѕР±СЂР°",
                "model_version": "1.0.0"
            }

        logger.info(f"РџРµСЂРµРґР±Р°С‡РµРЅРЅСЏ Р·Р°РІРµСЂС€РµРЅРѕ Р·Р° {time.time() - start_time:.2f} СЃРµРєСѓРЅРґ")

        return {
            "threats": mock_predictions,
            "recommendations": recommendations,
            "details": details
        }


