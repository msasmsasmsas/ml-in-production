#!/usr/bin/env python3

"""
Скрипт для завантаження тестового зображення для інференсу
"""

import requests
from PIL import Image
from io import BytesIO
import os
import sys

def download_test_image(output_path="tests/test_image.jpg", url="https://github.com/pytorch/hub/raw/master/images/dog.jpg"):
    # Створюємо директорію, якщо вона не існує
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    try:
        # Завантаження тестового зображення
        print(f"Завантаження зображення з {url}...")
        response = requests.get(url)
        response.raise_for_status()  # Перевірка на помилки HTTP

        img = Image.open(BytesIO(response.content))
        img.save(output_path)
        print(f"Зображення збережено як {output_path}")
        print(f"Розмір: {img.size[0]}x{img.size[1]}")
        return True
    except Exception as e:
        print(f"Помилка при завантаженні зображення: {e}")
        return False

if __name__ == "__main__":
    output_path = "tests/test_image.jpg"
    if len(sys.argv) > 1:
        output_path = sys.argv[1]

    success = download_test_image(output_path)
    if not success:
        sys.exit(1)
