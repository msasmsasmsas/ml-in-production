#!/usr/bin/env python3

"""
Приклад використання клієнта для Ray Serve API.
"""

import os
import sys
import time

# Додавання шляху до кореневого каталогу проекту
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from client.client import RayServeClient

def main():
    # Створення клієнта
    print("Підключення до API...")
    client = RayServeClient(base_url="http://localhost:8000")

    # Перевірка стану сервера
    try:
        health_status = client.health_check()
        print(f"Статус сервера: {health_status}")
    except Exception as e:
        print(f"Помилка при перевірці стану сервера: {e}")
        print("Переконайтеся, що сервер запущений за адресою http://localhost:8000")
        return

    # Отримання метаданих
    try:
        metadata = client.get_metadata()
        print("\nМетадані сервісу:")
        print(f"  Сервіс: {metadata['service']}")
        print(f"  Версія: {metadata['version']}")
        print(f"  Фреймворки: {', '.join(metadata['frameworks'])}")
    except Exception as e:
        print(f"Помилка при отриманні метаданих: {e}")

    # Класифікація зображення
    image_path = input("\nВведіть шлях до зображення для класифікації (або натисніть Enter для пропуску): ").strip()
    if image_path:
        try:
            print("\nВідправлення зображення на класифікацію...")
            start_time = time.time()
            result = client.predict(image_path, top_k=5)
            total_time = time.time() - start_time

            print(f"\nРезультати класифікації (отримано за {total_time:.2f} с):")
            print(f"Модель: {result['model_name']} (версія {result['model_version']})")
            print(f"Час інференсу: {result['inference_time']} с")

            print("\nТоп-5 класів:")
            for i, pred in enumerate(result['predictions']):
                print(f"  {i+1}. {pred['class_name']} ({pred['probability']:.4f})")
        except Exception as e:
            print(f"Помилка при класифікації зображення: {e}")

    print("\nРобота з клієнтом завершена.")

if __name__ == "__main__":
    main()
