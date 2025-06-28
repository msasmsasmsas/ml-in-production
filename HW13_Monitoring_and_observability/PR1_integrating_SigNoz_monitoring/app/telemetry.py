#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Модуль для налаштування телеметрії OpenTelemetry для SigNoz
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
    Налаштування телеметрії OpenTelemetry для SigNoz

    Args:
        service_name (str, optional): Назва сервісу. За замовчуванням використовує налаштування з конфігурації.
    """
    if not settings.OTEL_ENABLED:
        logger.info("Телеметрія OpenTelemetry вимкнена")
        return

    service_name = service_name or settings.OTEL_SERVICE_NAME

    logger.info(f"Налаштування телеметрії OpenTelemetry для сервісу '{service_name}'")

    # Створюємо ресурс з атрибутами сервісу
    resource_attributes = {}
    for attr_pair in settings.OTEL_RESOURCE_ATTRIBUTES.split(","):
        if "=" in attr_pair:
            key, value = attr_pair.split("=", 1)
            resource_attributes[key.strip()] = value.strip()

    resource = Resource.create({
        SERVICE_NAME: service_name,
        **resource_attributes
    })

    # Створюємо TracerProvider з нашим ресурсом
    tracer_provider = TracerProvider(resource=resource)

    # Створюємо експортер для SigNoz
    otlp_exporter = OTLPSpanExporter(
        endpoint=settings.OTEL_EXPORTER_OTLP_ENDPOINT,
        insecure=True
    )

    # Додаємо експортер до TracerProvider
    span_processor = BatchSpanProcessor(otlp_exporter)
    tracer_provider.add_span_processor(span_processor)

    # Встановлюємо TracerProvider глобально
    trace.set_tracer_provider(tracer_provider)

    # Ініціалізуємо автоматичну інструментацію
    RequestsInstrumentor().instrument()
    LoggingInstrumentor().instrument()

    logger.info("Телеметрія OpenTelemetry успішно налаштована")

    return tracer_provider

def instrument_fastapi(app):
    """
    Інструментує FastAPI додаток для збору телеметрії

    Args:
        app: Екземпляр FastAPI додатку
    """
    if not settings.OTEL_ENABLED:
        logger.info("Інструментація FastAPI вимкнена, оскільки телеметрія вимкнена")
        return

    logger.info("Інструментація FastAPI додатку для телеметрії")
    FastAPIInstrumentor.instrument_app(app, tracer_provider=trace.get_tracer_provider())

def create_custom_span(name, attributes=None):
    """
    Створює кастомний спан для ручного трекінгу

    Args:
        name (str): Назва спану
        attributes (dict, optional): Додаткові атрибути для спану

    Returns:
        Span: Об'єкт спану OpenTelemetry
    """
    tracer = trace.get_tracer(__name__)
    attributes = attributes or {}
    return tracer.start_as_current_span(name, attributes=attributes)
