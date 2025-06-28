#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Модуль для роботи з моделлю виявлення загроз з інтеграцією метрик
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
    Клас для роботи з моделлю виявлення загроз сільськогосподарським культурам
    з інтеграцією метрик для Prometheus
    """
    def __init__(self, model_path: str = None):
        """
        Ініціалізація моделі

        Args:
            model_path (str, optional): Шлях до файлу моделі
        """
        self.model_path = model_path or settings.MODEL_PATH
        logger.info(f"Ініціалізація моделі з {self.model_path}")

        # У реальній імплементації тут завантажуємо модель
        # self.model = torch.load(self.model_path) if os.path.exists(self.model_path) else None
        self.model = None
        self.class_names = [
            "healthy", "late_blight", "early_blight", "rust", 
            "septoria", "aphids", "thrips", "whitefly", 
            "bindweed", "nutsedge", "chickweed"
        ]
        logger.info("Модель успішно ініціалізована")

    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """
        Попередня обробка зображення для моделі

        Args:
            image (Image.Image): Вхідне зображення

        Returns:
            torch.Tensor: Підготовлений тензор для моделі
        """
        logger.debug("Попередня обробка зображення")
        img_array = np.array(image.convert("RGB").resize((224, 224)))
        # Конвертуємо в тензор, нормалізуємо, тощо
        tensor = torch.tensor(img_array).float().permute(2, 0, 1) / 255.0
        return tensor.unsqueeze(0)

    @prediction_processing_time.time()
    def predict(self, image: Image.Image, confidence_threshold: float = 0.5) -> Dict[str, Any]:
        """
        Обробка зображення та повернення передбачень

        Args:
            image (Image.Image): Вхідне зображення
            confidence_threshold (float, optional): Поріг впевненості для виявлення загроз

        Returns:
            Dict[str, Any]: Результати передбачення
        """
        logger.info(f"Виконуємо передбачення з порогом впевненості: {confidence_threshold}")
        start_time = time.time()

        # Попередня обробка зображення
        tensor = self.preprocess_image(image)

        # У реальній імплементації запускаємо модель
        # predictions = self.model(tensor)

        # Мокові передбачення для тестування
        # Робимо передбачення детерміністичними на основі вмісту зображення
        np.random.seed(int(tensor.sum().item() * 100))

        mock_predictions = []
        # Генеруємо 0-3 випадкових загроз
        num_threats = np.random.randint(0, 4)

        for _ in range(num_threats):
            threat_type = np.random.choice(["disease", "pest", "weed"])
            confidence = np.random.uniform(0.3, 0.95)

            # Реєструємо рівень впевненості
            prediction_confidence.observe(confidence)

            # Пропускаємо, якщо нижче порогу
            if confidence < confidence_threshold:
                continue

            # Вибираємо назву на основі типу
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

        # Оновлюємо лічильники передбачень
        if mock_predictions:
            prediction_count.labels(result="positive").inc()
        else:
            prediction_count.labels(result="negative").inc()

        # Генеруємо рекомендації на основі виявлених загроз
        recommendations = []
        details = {}

        if not mock_predictions:
            recommendations.append("Загроз не виявлено. Продовжуйте регулярний моніторинг посівів.")
        else:
            for pred in mock_predictions:
                if pred["type"] == "disease":
                    recommendations.append(f"Для {pred['name']}, розгляньте застосування відповідного фунгіциду.")
                elif pred["type"] == "pest":
                    recommendations.append(f"Для {pred['name']}, розгляньте використання інсектициду або біологічного контролю.")
                else:  # weed
                    recommendations.append(f"Для {pred['name']}, розгляньте механічне видалення або цільовий гербіцид.")

            # Додаємо деякі деталі
            details = {
                "severity": "низька" if len(mock_predictions) == 1 else "середня" if len(mock_predictions) == 2 else "висока",
                "processing_time_ms": round((time.time() - start_time) * 1000),
                "image_quality": "добра",
                "model_version": "1.0.0"
            }

        logger.info(f"Передбачення завершено за {time.time() - start_time:.2f} секунд")

        return {
            "threats": mock_predictions,
            "recommendations": recommendations,
            "details": details
        }
