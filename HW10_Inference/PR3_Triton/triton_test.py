#!/usr/bin/env python3

"""
Простий скрипт для тестування підключення до Triton Inference Server.
Цей скрипт перевіряє, чи доступний сервер і виводить інформацію про нього.
"""

import sys
import time
import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException

def test_server_connection(url='localhost:8000', timeout_seconds=30):
    """Перевіряє підключення до Triton сервера з періодичними спробами протягом вказаного часу."""
    print(f"Перевірка підключення до Triton сервера за адресою {url}...")

    start_time = time.time()
    attempt = 1

    while time.time() - start_time < timeout_seconds:
        try:
            # Створюємо клієнт
            client = httpclient.InferenceServerClient(url=url)

            # Перевіряємо готовність сервера
            if client.is_server_ready():
                print(f"✅ Triton сервер готовий (спроба {attempt})")

                # Отримуємо та виводимо метадані сервера
                metadata = client.get_server_metadata()
                print(f"\nІнформація про сервер:")
                print(f"Назва: {metadata['name']}")
                print(f"Версія: {metadata['version']}")

                # Отримуємо доступні моделі
                models = client.get_model_repository_index()
                print(f"\nДоступні моделі:")
                if models:
                    for model in models:
                        print(f" - {model['name']}")
                        try:
                            if client.is_model_ready(model['name']):
                                print(f"   ✅ Модель готова")

                                # Виводимо метадані моделі
                                model_metadata = client.get_model_metadata(model['name'])
                                print(f"   Платформа: {model_metadata.get('platform', 'Не вказана')}")
                                print(f"   Входи: {len(model_metadata.get('inputs', []))}")
                                print(f"   Виходи: {len(model_metadata.get('outputs', []))}")
                            else:
                                print(f"   ❌ Модель не готова")
                        except Exception as e:
                            print(f"   ❌ Помилка перевірки моделі: {e}")
                else:
                    print("Моделі не знайдені. Переконайтеся, що директорія model_repository містить моделі.")

                return True
            else:
                print(f"⏳ Triton сервер не готовий (спроба {attempt})")
        except InferenceServerException as e:
            print(f"⏳ Помилка підключення (спроба {attempt}): {e}")
        except Exception as e:
            print(f"⏳ Несподівана помилка (спроба {attempt}): {e}")

        # Збільшуємо лічильник спроб та чекаємо перед наступною спробою
        attempt += 1
        time.sleep(1)

    print(f"❌ Не вдалося підключитися до Triton сервера протягом {timeout_seconds} секунд")
    return False

if __name__ == "__main__":
    # Використовуємо аргумент командного рядка як URL сервера, якщо він вказаний
    server_url = sys.argv[1] if len(sys.argv) > 1 else 'localhost:8000'

    # Запускаємо перевірку
    success = test_server_connection(server_url)

    # Виходимо з відповідним кодом
    sys.exit(0 if success else 1)
