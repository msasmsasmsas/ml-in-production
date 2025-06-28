#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Скрипт для локального запуску сервера класифікації без Docker
"""

import argparse
import sys
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Запуск локального сервера класифікації")
    parser.add_argument("--port", type=int, default=9000, help="Порт для сервера (за замовчуванням: 9000)")
    args = parser.parse_args()

    # Встановлення змінних середовища
    os.environ["MODEL_NAME"] = "model.ResNet50Classifier"
    os.environ["SERVICE_TYPE"] = "MODEL"
    os.environ["PERSISTENCE"] = "0"

    # Запуск сервера Seldon
    logger.info(f"Запуск сервера на порту {args.port}...")
    try:
        from seldon_core.microservice import start_servers
        start_servers()
    except Exception as e:
        logger.error(f"Помилка запуску сервера: {e}")
        sys.exit(1)

    logger.info("Сервер запущено.")

if __name__ == "__main__":
    main()
