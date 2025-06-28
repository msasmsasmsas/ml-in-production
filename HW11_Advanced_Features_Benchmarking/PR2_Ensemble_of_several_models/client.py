import requests
import time
import argparse
import sys
import os
from pathlib import Path
import json
import concurrent.futures

class EnsembleClient:
    """
    Клієнт для тестування сервера з ансамблем моделей
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

    def predict(self, image_path, include_individual=False):
        """
        Відправляє запит на прогнозування

        Параметри:
        -----------
        image_path: шлях до файлу зображення
        include_individual: прапорець для включення індивідуальних прогнозів моделей

        Повертає:
        -----------
        словник з результатами прогнозування або помилкою
        """
        try:
            with open(image_path, 'rb') as f:
                files = {'file': (os.path.basename(image_path), f, 'image/jpeg')}

                params = {}
                if include_individual:
                    params['include_individual'] = 'true'

                start_time = time.time()
                response = requests.post(self.predict_url, files=files, params=params)
                elapsed = time.time() - start_time

                if response.status_code == 200:
                    result = response.json()
                    result['latency'] = elapsed
                    return result
                else:
                    return {'error': f'HTTP помилка: {response.status_code}', 'response': response.text, 'latency': elapsed}
        except Exception as e:
            return {'error': str(e), 'latency': 0}

def pretty_print_predictions(predictions):
    """
    Виводить прогнози у форматованому вигляді

    Параметри:
    -----------
    predictions: список прогнозів
    """
    for i, pred in enumerate(predictions):
        print(f"{i+1}. {pred['class_name']} ({pred['class_id']}) - {pred['score']*100:.2f}%")

def run_ensemble_test(client, image_path):
    """
    Запускає тестування ансамблю моделей

    Параметри:
    -----------
    client: екземпляр класу EnsembleClient
    image_path: шлях до файлу зображення
    """
    print(f"Тестування ансамблю моделей для зображення: {image_path}")

    # Запит з індивідуальними прогнозами
    print("\nЗапит з індивідуальними прогнозами:")
    result = client.predict(image_path, include_individual=True)

    if 'error' in result:
        print(f"Помилка: {result['error']}")
        if 'response' in result:
            print(f"Відповідь сервера: {result['response']}")
        return

    print(f"\nРезультати ансамблю (метод: {result['aggregation_method']}):\n")
    pretty_print_predictions(result['ensemble_predictions'])

    print("\nРезультати окремих моделей:")
    for model_result in result['individual_predictions']:
        print(f"\nМодель: {model_result['model_name']} (вага: {model_result['weight']})")
        pretty_print_predictions(model_result['predictions'])

    print(f"\nЗагальний час запиту: {result['latency']*1000:.2f} мс")

def run_concurrent_test(client, image_path, num_requests, concurrency):
    """
    Запускає конкурентне тестування сервера з ансамблем моделей

    Параметри:
    -----------
    client: екземпляр класу EnsembleClient
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
    parser = argparse.ArgumentParser(description='Клієнт для тестування сервера з ансамблем моделей')
    parser.add_argument('--url', type=str, default='http://localhost:5000', help='Базова URL сервера')
    parser.add_argument('--image', type=str, default='test_image.jpg', help='Шлях до тестового зображення')
    parser.add_argument('--mode', type=str, choices=['detail', 'benchmark'], default='detail', 
                        help='Режим роботи: детальний аналіз (detail) або тестування продуктивності (benchmark)')
    parser.add_argument('--requests', type=int, default=10, help='Кількість запитів для режиму benchmark')
    parser.add_argument('--concurrency', type=int, default=2, help='Рівень паралелізму для режиму benchmark')

    args = parser.parse_args()

    if not Path(args.image).exists():
        print(f"Помилка: файл {args.image} не існує")
        sys.exit(1)

    client = EnsembleClient(args.url)

    # Перевірка здоров'я сервера
    try:
        health_response = requests.get(f"{args.url}/health")
        if health_response.status_code != 200:
            print(f"Сервер не готовий до роботи. Статус: {health_response.status_code}")
            sys.exit(1)

        health_data = health_response.json()
        print(f"Сервер готовий до роботи. Завантажено {health_data['models_loaded']} моделей.")
        print(f"Метод агрегації: {health_data['aggregation_method']}")

    except Exception as e:
        print(f"Не вдалося підключитися до сервера: {e}")
        sys.exit(1)

    if args.mode == 'detail':
        run_ensemble_test(client, args.image)
    else:  # benchmark
        run_concurrent_test(client, args.image, args.requests, args.concurrency)
