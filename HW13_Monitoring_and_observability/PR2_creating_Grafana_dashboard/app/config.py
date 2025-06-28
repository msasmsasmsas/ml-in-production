# новлена версія для PR
# новлена версія для PR
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
РљРѕРЅС„С–РіСѓСЂР°С†С–СЏ РґРѕРґР°С‚РєСѓ С‚Р° РЅР°Р»Р°С€С‚СѓРІР°РЅРЅСЏ РґР»СЏ РјРѕРЅС–С‚РѕСЂРёРЅРіСѓ
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

    # РќР°Р»Р°С€С‚СѓРІР°РЅРЅСЏ РґР»СЏ РјРѕРЅС–С‚РѕСЂРёРЅРіСѓ
    METRICS_PORT: int = int(os.getenv("METRICS_PORT", "8000"))

    # РќР°Р»Р°С€С‚СѓРІР°РЅРЅСЏ РґР»СЏ РјРѕРґРµР»С–
    MODEL_PATH: str = os.getenv("MODEL_PATH", "models/threat_detection_model.pt")

    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()


