# новлена версія для PR
# новлена версія для PR
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
РњРѕРґСѓР»СЊ РґР»СЏ РЅР°Р»Р°С€С‚СѓРІР°РЅРЅСЏ С‚РµР»РµРјРµС‚СЂС–С— OpenTelemetry РґР»СЏ SigNoz
"""

import logging
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.resources import Resource
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.instrumentation.logging import LoggingInstrumentor
from opentelemetry.sdk.resources import SERVICE_NAME, Resource

from app.config import settings

logger = logging.getLogger(__name__)

def setup_telemetry(service_name=None):
    """
    РќР°Р»Р°С€С‚СѓРІР°РЅРЅСЏ С‚РµР»РµРјРµС‚СЂС–С— OpenTelemetry РґР»СЏ SigNoz

    Args:
        service_name (str, optional): РќР°Р·РІР° СЃРµСЂРІС–СЃСѓ. Р—Р° Р·Р°РјРѕРІС‡СѓРІР°РЅРЅСЏРј РІРёРєРѕСЂРёСЃС‚РѕРІСѓС” РЅР°Р»Р°С€С‚СѓРІР°РЅРЅСЏ Р· РєРѕРЅС„С–РіСѓСЂР°С†С–С—.
    """
    if not settings.OTEL_ENABLED:
        logger.info("РўРµР»РµРјРµС‚СЂС–СЏ OpenTelemetry РІРёРјРєРЅРµРЅР°")
        return

    service_name = service_name or settings.OTEL_SERVICE_NAME

    logger.info(f"РќР°Р»Р°С€С‚СѓРІР°РЅРЅСЏ С‚РµР»РµРјРµС‚СЂС–С— OpenTelemetry РґР»СЏ СЃРµСЂРІС–СЃСѓ '{service_name}'")

    # РЎС‚РІРѕСЂСЋС”РјРѕ СЂРµСЃСѓСЂСЃ Р· Р°С‚СЂРёР±СѓС‚Р°РјРё СЃРµСЂРІС–СЃСѓ
    resource_attributes = {}
    for attr_pair in settings.OTEL_RESOURCE_ATTRIBUTES.split(","):
        if "=" in attr_pair:
            key, value = attr_pair.split("=", 1)
            resource_attributes[key.strip()] = value.strip()

    resource = Resource.create({
        SERVICE_NAME: service_name,
        **resource_attributes
    })

    # РЎС‚РІРѕСЂСЋС”РјРѕ TracerProvider Р· РЅР°С€РёРј СЂРµСЃСѓСЂСЃРѕРј
    tracer_provider = TracerProvider(resource=resource)

    # РЎС‚РІРѕСЂСЋС”РјРѕ РµРєСЃРїРѕСЂС‚РµСЂ РґР»СЏ SigNoz
    otlp_exporter = OTLPSpanExporter(
        endpoint=settings.OTEL_EXPORTER_OTLP_ENDPOINT,
        insecure=True
    )

    # Р”РѕРґР°С”РјРѕ РµРєСЃРїРѕСЂС‚РµСЂ РґРѕ TracerProvider
    span_processor = BatchSpanProcessor(otlp_exporter)
    tracer_provider.add_span_processor(span_processor)

    # Р’СЃС‚Р°РЅРѕРІР»СЋС”РјРѕ TracerProvider РіР»РѕР±Р°Р»СЊРЅРѕ
    trace.set_tracer_provider(tracer_provider)

    # Р†РЅС–С†С–Р°Р»С–Р·СѓС”РјРѕ Р°РІС‚РѕРјР°С‚РёС‡РЅСѓ С–РЅСЃС‚СЂСѓРјРµРЅС‚Р°С†С–СЋ
    RequestsInstrumentor().instrument()
    LoggingInstrumentor().instrument()

    logger.info("РўРµР»РµРјРµС‚СЂС–СЏ OpenTelemetry СѓСЃРїС–С€РЅРѕ РЅР°Р»Р°С€С‚РѕРІР°РЅР°")

    return tracer_provider

def instrument_fastapi(app):
    """
    Р†РЅСЃС‚СЂСѓРјРµРЅС‚СѓС” FastAPI РґРѕРґР°С‚РѕРє РґР»СЏ Р·Р±РѕСЂСѓ С‚РµР»РµРјРµС‚СЂС–С—

    Args:
        app: Р•РєР·РµРјРїР»СЏСЂ FastAPI РґРѕРґР°С‚РєСѓ
    """
    if not settings.OTEL_ENABLED:
        logger.info("Р†РЅСЃС‚СЂСѓРјРµРЅС‚Р°С†С–СЏ FastAPI РІРёРјРєРЅРµРЅР°, РѕСЃРєС–Р»СЊРєРё С‚РµР»РµРјРµС‚СЂС–СЏ РІРёРјРєРЅРµРЅР°")
        return

    logger.info("Р†РЅСЃС‚СЂСѓРјРµРЅС‚Р°С†С–СЏ FastAPI РґРѕРґР°С‚РєСѓ РґР»СЏ С‚РµР»РµРјРµС‚СЂС–С—")
    FastAPIInstrumentor.instrument_app(app, tracer_provider=trace.get_tracer_provider())

def create_custom_span(name, attributes=None):
    """
    РЎС‚РІРѕСЂСЋС” РєР°СЃС‚РѕРјРЅРёР№ СЃРїР°РЅ РґР»СЏ СЂСѓС‡РЅРѕРіРѕ С‚СЂРµРєС–РЅРіСѓ

    Args:
        name (str): РќР°Р·РІР° СЃРїР°РЅСѓ
        attributes (dict, optional): Р”РѕРґР°С‚РєРѕРІС– Р°С‚СЂРёР±СѓС‚Рё РґР»СЏ СЃРїР°РЅСѓ

    Returns:
        Span: РћР±'С”РєС‚ СЃРїР°РЅСѓ OpenTelemetry
    """
    tracer = trace.get_tracer(__name__)
    attributes = attributes or {}
    return tracer.start_as_current_span(name, attributes=attributes)


