import requests
import time
import sys
import concurrent.futures
import argparse
import os
from pathlib import Path

class ModelClient:
    """
    Клієнт для тестування сервера з динамічним пакетуванням
    """
    def __init__(self, base_url):
        """
        Ініціалізація клієнта

        Параметри:
        -----------
        base_url: базова URL адреса сервера
        """
        self.base_url = base_url
        self.predict_url = f"{base_url}/predict"

    def predict(self, image_path):
        """
        Відправляє запит на прогнозування

        Параметри:
        -----------
        image_path: шлях до файлу зображення

        Повертає:
        -----------
        словник з результатами прогнозування або помилкою
        """
        try:
            with open(image_path, 'rb') as f:
                files = {'file': (os.path.basename(image_path), f, 'image/jpeg')}
                start_time = time.time()
                response = requests.post(self.predict_url, files=files)
                elapsed = time.time() - start_time

                if response.status_code == 200:
                    result = response.json()
                    result['latency'] = elapsed
                    return result
                else:
                    return {'error': f'HTTP помилка: {response.status_code}', 'response': response.text, 'latency': elapsed}
        except Exception as e:
            return {'error': str(e), 'latency': 0}

def run_concurrent_test(client, image_path, num_requests, concurrency):
    """
    Запускає конкурентне тестування сервера

    Параметри:
    -----------
    client: екземпляр класу ModelClient
    image_path: шлях до файлу зображення
    num_requests: загальна кількість запитів
    concurrency: кількість паралельних запитів
    """
    print(f"Запуск {num_requests} запитів з рівнем паралелізму {concurrency}")

    results = []
    errors = 0

    start_time = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        future_to_id = {executor.submit(client.predict, image_path): i for i in range(num_requests)}

        for future in concurrent.futures.as_completed(future_to_id):
            result = future.result()
            if 'error' in result:
                errors += 1
            results.append(result)

    total_time = time.time() - start_time

    # Обчислення статистики
    latencies = [r['latency'] for r in results]
    avg_latency = sum(latencies) / len(latencies)
    min_latency = min(latencies)
    max_latency = max(latencies)
    p95_latency = sorted(latencies)[int(len(latencies) * 0.95)]
    rps = num_requests / total_time

    print(f"\nРезультати тестування:")
    print(f"Загальний час: {total_time:.2f} с")
    print(f"Успішних запитів: {num_requests - errors} з {num_requests} ({100 * (num_requests - errors) / num_requests:.2f}%)")
    print(f"RPS (запитів на секунду): {rps:.2f}")
    print(f"Середня затримка: {avg_latency * 1000:.2f} мс")
    print(f"Мінімальна затримка: {min_latency * 1000:.2f} мс")
    print(f"Максимальна затримка: {max_latency * 1000:.2f} мс")
    print(f"P95 затримка: {p95_latency * 1000:.2f} мс")

    return {
        'total_time': total_time,
        'successful_requests': num_requests - errors,
        'total_requests': num_requests,
        'rps': rps,
        'avg_latency_ms': avg_latency * 1000,
        'min_latency_ms': min_latency * 1000,
        'max_latency_ms': max_latency * 1000,
        'p95_latency_ms': p95_latency * 1000
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Клієнт для тестування сервера з динамічним пакетуванням')
    parser.add_argument('--url', type=str, default='http://localhost:5000', help='Базова URL сервера')
    parser.add_argument('--image', type=str, default='test_image.jpg', help='Шлях до тестового зображення')
    parser.add_argument('--requests', type=int, default=100, help='Кількість запитів')
    parser.add_argument('--concurrency', type=int, default=10, help='Рівень паралелізму')

    args = parser.parse_args()

    if not Path(args.image).exists():
        print(f"Помилка: файл {args.image} не існує")
        sys.exit(1)

    client = ModelClient(args.url)

    # Перевірка здоров'я сервера
    try:
        health_response = requests.get(f"{args.url}/health")
        if health_response.status_code != 200:
            print(f"Сервер не готовий до роботи. Статус: {health_response.status_code}")
            sys.exit(1)
    except Exception as e:
        print(f"Не вдалося підключитися до сервера: {e}")
        sys.exit(1)

    print(f"Сервер готовий до роботи. Початок тестування...")
    run_concurrent_test(client, args.image, args.requests, args.concurrency)
