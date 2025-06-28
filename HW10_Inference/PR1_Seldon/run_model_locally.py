#!/usr/bin/env python3
"""
Запуск микросервиса Seldon Core локально без Docker.
"""

import sys
import os
import logging
import argparse
from PIL import Image
from model.ResNet50Classifier import ResNet50Classifier

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_microservice():
    """Запуск мікросервісу Seldon Core"""
    try:
        from seldon_core.microservice import get_rest_microservice
        from seldon_core.seldon_client import SeldonClient

        logger.info("Ініціалізація моделі ResNet50Classifier")
        model = ResNet50Classifier()

        app = get_rest_microservice(model)

        logger.info("Запуск мікросервісу Seldon на порту 9000")
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=9000)
    except ImportError as e:
        logger.error(f"Помилка імпорту модулів Seldon Core: {e}")
        logger.info("Запуск в автономному режимі для тестування моделі")
        test_model()

def test_model(image_path=None):
    """Тестирование модели без микросервиса"""
    try:
        # Инициализация модели
        logger.info("Инициализация модели ResNet50Classifier для тестирования")
        model = ResNet50Classifier()

        # Если путь к изображению не указан, используем стандартное тестовое изображение
        if not image_path:
            # Поиск тестового изображения
            test_dirs = ["tests", "test", "./", "../"]
            image_types = ["test_image.jpg", "test.jpg", "sample.jpg", "image.jpg"]

            for test_dir in test_dirs:
                for img_type in image_types:
                    path = os.path.join(test_dir, img_type)
                    if os.path.exists(path):
                        image_path = path
                        break
                if image_path:
                    break

            if not image_path:
                logger.error("Не найдено тестовое изображение")
                # Создаем пустое изображение 224x224 для тестирования
                img = Image.new('RGB', (224, 224), color='white')
                tmp_path = 'temp_test_image.jpg'
                img.save(tmp_path)
                image_path = tmp_path
                logger.info(f"Создано временное тестовое изображение: {image_path}")

        logger.info(f"Тестирование модели с изображением: {image_path}")
        result = model.predict(image_path)

        # Вывод результатов
        print("\nРезультаты предсказания:")
        print(f"Модель: {result['model_name']} (версия {result['model_version']})")
        print("\nТоп предсказаний:")

        for pred in result["predictions"]:
            print(f"{pred['rank']}. {pred['class_name']}: {pred['probability']:.4f}")

        # Удаляем временный файл если он был создан
        if image_path == 'temp_test_image.jpg' and os.path.exists(image_path):
            os.remove(image_path)

    except Exception as e:
        logger.error(f"Ошибка при тестировании модели: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Запуск микросервиса Seldon Core или тестирование модели")
    parser.add_argument("--test", action="store_true", help="Только тестирование модели без запуска микросервиса")
    parser.add_argument("--image", type=str, help="Путь к изображению для тестирования")

    args = parser.parse_args()

    if args.test:
        test_model(args.image)
    else:
        run_microservice()
