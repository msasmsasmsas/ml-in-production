#!/usr/bin/env python3

"""
Простий тест для перевірки роботи Ray Serve.
"""

import requests

# Перевірка ендпоінту /health
try:
    response = requests.get("http://localhost:8000/health")
    print(f"Статус код: {response.status_code}")
    print(f"Відповідь: {response.json()}")
    if response.status_code == 200:
        print("Тест пройдено успішно!")
    else:
        print("Тест не пройдено: отримано помилковий статус код.")
except Exception as e:
    print(f"Тест не пройдено: виникла помилка: {str(e)}")
