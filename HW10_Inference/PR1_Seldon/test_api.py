#!/usr/bin/env python3
"""
Тестування API мікросервісу Seldon Core.
"""

import requests
import json
import argparse
import numpy as np
from PIL import Image
import os
import logging

# Налаштування логування
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_seldon_api(image_path, host="localhost", port=9000):
    """Тестування API Seldon"""
    try:
        # Завантаження зображення
        logger.info(f"Завантаження зображення: {image_path}")
        image = Image.open(image_path).convert('RGB')

        # Перетворення зображення в масив
        image = image.resize((224, 224))
        image_array = np.array(image)

        # Формування запиту
        payload = {
            "data": {
                "ndarray": image_array.tolist()
            }
        }

        # URL для запиту
        url = f"http://{host}:{port}/api/v1.0/predictions"
        logger.info(f"Надсилання запиту на: {url}")

        # Надсилання запиту з обробкою помилок
        try:
            response = requests.post(url, json=payload, timeout=10)
        except requests.exceptions.ConnectionError:
            # Пробуємо альтернативний URL для FastAPI-сервера
            logger.warning(f"Не вдалося підключитися до {url}, пробуємо альтернативний URL /predict")
            url = f"http://{host}:{port}/predict"
            logger.info(f"Надсилання запиту на: {url}")

            # Надсилаємо як multipart/form-data
            files = {"file": ("image.jpg", open(image_path, "rb"), "image/jpeg")}
            data = {"confidence": "0.0"}
            response = requests.post(url, files=files, data=data, timeout=10)

        # Перевірка статусу відповіді
        if response.status_code == 200:
            result = response.json()

            # Виведення результатів
            print("\nРезультати API запиту:")
            print(json.dumps(result, indent=2))
            print("\nТоп прогнозувань:")

            if "predictions" in result:
                for pred in result["predictions"]:
                    print(f"{pred['rank']}. {pred['class_name']}: {pred['probability']:.4f}")
            else:
                print("Структура відповіді не містить прогнозувань")
        else:
            print(f"Помилка запиту: {response.status_code}")
            print(response.text)

    except Exception as e:
        logger.error(f"Помилка при тестуванні API: {e}")
        import traceback
        traceback.print_exc()

def find_test_image():
    """Поиск тестового изображения"""
    test_dirs = ["tests", "test", "./", "../"]
    image_types = ["test_image.jpg", "test.jpg", "sample.jpg", "image.jpg"]

    for test_dir in test_dirs:
        for img_type in image_types:
            path = os.path.join(test_dir, img_type)
            if os.path.exists(path):
                return path

    # Если изображение не найдено, создаем временное
    logger.warning("Тестовое изображение не найдено, создаем временное")
    img = Image.new('RGB', (224, 224), color='white')
    tmp_path = 'temp_test_image.jpg'
    img.save(tmp_path)
    return tmp_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Тестирование API Seldon Core")
    parser.add_argument("--image", type=str, help="Путь к изображению для тестирования")
    parser.add_argument("--host", type=str, default="localhost", help="Хост API (по умолчанию: localhost)")
    parser.add_argument("--port", type=int, default=9000, help="Порт API (по умолчанию: 9000)")

    args = parser.parse_args()

    # Если путь к изображению не указан, ищем его
    if not args.image:
        args.image = find_test_image()

    test_seldon_api(args.image, args.host, args.port)

    # Удаляем временный файл если он был создан
    if args.image == 'temp_test_image.jpg' and os.path.exists(args.image):
        os.remove(args.image)
