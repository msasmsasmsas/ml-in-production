# новлена версія для PR
# новлена версія для PR
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
РљРѕРЅС„С–РіСѓСЂР°С†С–СЏ РґРѕРґР°С‚РєСѓ С‚Р° РЅР°Р»Р°С€С‚СѓРІР°РЅРЅСЏ РґР»СЏ РјРѕРЅС–С‚РѕСЂРёРЅРіСѓ SigNoz
"""

import os
from pydantic import BaseSettings

class Settings(BaseSettings):
    """
    РќР°Р»Р°С€С‚СѓРІР°РЅРЅСЏ РґРѕРґР°С‚РєСѓ Р· СЃРµСЂРІС–СЃРѕРј СЂРѕР·РїС–Р·РЅР°РІР°РЅРЅСЏ Р·Р°РіСЂРѕР·
    """
    # РћСЃРЅРѕРІРЅС– РЅР°Р»Р°С€С‚СѓРІР°РЅРЅСЏ API
    API_VERSION: str = "1.0.0"
    DEBUG: bool = os.getenv("DEBUG", "False").lower() in ("true", "1", "t")

    # РќР°Р»Р°С€С‚СѓРІР°РЅРЅСЏ РґР»СЏ OpenTelemetry/SigNoz
    OTEL_ENABLED: bool = os.getenv("OTEL_ENABLED", "True").lower() in ("true", "1", "t")
    OTEL_SERVICE_NAME: str = os.getenv("OTEL_SERVICE_NAME", "threat-detection-api")
    OTEL_EXPORTER_OTLP_ENDPOINT: str = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://signoz:4317")
    OTEL_RESOURCE_ATTRIBUTES: str = os.getenv(
        "OTEL_RESOURCE_ATTRIBUTES", 
        "deployment.environment=development,service.version=1.0.0"
    )

    # РќР°Р»Р°С€С‚СѓРІР°РЅРЅСЏ РґР»СЏ РјРѕРґРµР»С–
    MODEL_PATH: str = os.getenv("MODEL_PATH", "models/threat_detection_model.pt")

    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()


