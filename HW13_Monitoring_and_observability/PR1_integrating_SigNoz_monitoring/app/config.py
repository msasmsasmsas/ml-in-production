#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Конфігурація додатку та налаштування для моніторингу SigNoz
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

    # Налаштування для OpenTelemetry/SigNoz
    OTEL_ENABLED: bool = os.getenv("OTEL_ENABLED", "True").lower() in ("true", "1", "t")
    OTEL_SERVICE_NAME: str = os.getenv("OTEL_SERVICE_NAME", "threat-detection-api")
    OTEL_EXPORTER_OTLP_ENDPOINT: str = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://signoz:4317")
    OTEL_RESOURCE_ATTRIBUTES: str = os.getenv(
        "OTEL_RESOURCE_ATTRIBUTES", 
        "deployment.environment=development,service.version=1.0.0"
    )

    # Налаштування для моделі
    MODEL_PATH: str = os.getenv("MODEL_PATH", "models/threat_detection_model.pt")

    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()
