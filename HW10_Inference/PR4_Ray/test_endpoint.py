#!/usr/bin/env python3

"""
Простий скрипт для тестування ендпоінтів Ray Serve.
"""

import requests
import argparse
import time
import sys

def test_health(base_url):
    """Тестування ендпоінту health."""
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        print(f"Status code: {response.status_code}")
        if response.status_code == 200:
            print(f"Response: {response.json()}")
            return True
        else:
            print(f"Помилка: {response.text}")
            return False
    except Exception as e:
        print(f"Помилка при з'єднанні: {str(e)}")
        return False

def test_metadata(base_url):
    """Тестування ендпоінту metadata."""
    try:
        response = requests.get(f"{base_url}/metadata", timeout=5)
        print(f"Status code: {response.status_code}")
        if response.status_code == 200:
            print(f"Response: {response.json()}")
            return True
        else:
            print(f"Помилка: {response.text}")
            return False
    except Exception as e:
        print(f"Помилка при з'єднанні: {str(e)}")
        return False

def wait_for_service(base_url, max_retries=5, retry_delay=2):
    """Очікування на запуск сервісу."""
    print(f"Очікування на запуск сервісу за адресою {base_url}...")
    for i in range(max_retries):
        if test_health(base_url):
            print("Сервіс готовий до роботи!")
            return True
        print(f"Спроба {i+1}/{max_retries} не вдалася. Очікування {retry_delay} секунд...")
        time.sleep(retry_delay)

    print("Не вдалося підключитися до сервісу після кількох спроб.")
    return False

def main():
    parser = argparse.ArgumentParser(description="Тестування ендпоінтів Ray Serve")
    parser.add_argument(
        "--url", type=str, default="http://localhost:8000", help="Base URL of the Ray Serve deployment"
    )
    parser.add_argument(
        "--wait", action="store_true", help="Wait for service to start"
    )
    args = parser.parse_args()

    base_url = args.url.rstrip("/")

    if args.wait:
        if not wait_for_service(base_url):
            sys.exit(1)

    # Тестування ендпоінтів
    print("\n=== Тестування ендпоінту health ===")
    test_health(base_url)

    print("\n=== Тестування ендпоінту metadata ===")
    test_metadata(base_url)

if __name__ == "__main__":
    main()
