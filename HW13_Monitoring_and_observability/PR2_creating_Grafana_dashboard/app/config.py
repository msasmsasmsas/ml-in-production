#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Конфігурація додатку та налаштування для моніторингу
"""

import os
from pydantic import BaseSettings

class Settings(BaseSettings):
    """
    Налаштування додатку з сервісом розпізнавання загроз
    """
    # Основні налаштування API
    API_VERSION: str = "1.0.0"
    DEBUG: bool = os.getenv("DEBUG", "False").lower() in ("true", "1", "t")

    # Налаштування для моніторингу
    METRICS_PORT: int = int(os.getenv("METRICS_PORT", "8000"))

    # Налаштування для моделі
    MODEL_PATH: str = os.getenv("MODEL_PATH", "models/threat_detection_model.pt")

    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()
