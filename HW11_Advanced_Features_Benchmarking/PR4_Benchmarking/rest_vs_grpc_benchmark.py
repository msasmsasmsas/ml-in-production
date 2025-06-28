#!/usr/bin/env python
"""
Скрипт для порівняльного бенчмаркінгу REST та gRPC інтерфейсів моделей машинного навчання
"""

import os
import time
import json
import argparse
import statistics
import concurrent.futures
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Для REST запитів
import requests

# Для gRPC запитів
try:
    import grpc
    import inference_pb2
    import inference_pb2_grpc
    grpc_available = True
except ImportError:
    grpc_available = False
    print("УВАГА: gRPC модулі не знайдено. gRPC бенчмаркінг буде недоступний.")

class RestClient:
    """
    Клієнт для REST API
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
        кортеж (відповідь, час виконання, успіх)
        """
        try:
            with open(image_path, 'rb') as f:
                files = {'file': (os.path.basename(image_path), f, 'image/jpeg')}

                start_time = time.time()
                response = requests.post(self.predict_url, files=files)
                elapsed = time.time() - start_time

                if response.status_code == 200:
                    result = response.json()
                    return result, elapsed, True
                else:
                    return {'error': f'HTTP помилка: {response.status_code}'}, elapsed, False
        except Exception as e:
            return {'error': str(e)}, 0, False

class GrpcClient:
    """
    Клієнт для gRPC API
    """
    def __init__(self, server_address):
        """
        Ініціалізація клієнта

        Параметри:
        -----------
        server_address: адреса gRPC сервера
        """
        if not grpc_available:
            raise ImportError("gRPC модулі недоступні")

        # Створення каналу з опціями для великих повідомлень
        channel_options = [
            ('grpc.max_send_message_length', 50 * 1024 * 1024),  # 50 MB
            ('grpc.max_receive_message_length', 50 * 1024 * 1024),  # 50 MB
        ]
        self.channel = grpc.insecure_channel(server_address, options=channel_options)
        self.stub = inference_pb2_grpc.InferenceServiceStub(self.channel)

    def predict(self, image_path):
        """
        Відправляє запит на прогнозування

        Параметри:
        -----------
        image_path: шлях до файлу зображення

        Повертає:
        -----------
        кортеж (відповідь, час виконання, успіх)
        """
        try:
            # Зчитування файлу зображення
            with open(image_path, 'rb') as f:
                image_data = f.read()

            # Створення запиту
            request = inference_pb2.PredictRequest(
                data=image_data,
                content_type='image/jpeg'
            )

            # Вимірювання часу виконання запиту
            start_time = time.time()
            response = self.stub.Predict(request)
            elapsed = time.time() - start_time

            # Конвертація response у словник для уніфікації з REST
            result = {
                'request_id': response.request_id,
                'success': response.success,
                'processing_time': response.processing_time
            }

            if response.success:
                result['predictions'] = []
                for pred in response.predictions:
                    result['predictions'].append({
                        'class_id': pred.class_id,
                        'class_name': pred.class_name,
                        'score': pred.score
                    })
                return result, elapsed, True
            else:
                result['error'] = response.error
                return result, elapsed, False

        except Exception as e:
            return {'error': str(e)}, 0, False

    def close(self):
        """
        Закриття з'єднання
        """
        self.channel.close()

def run_benchmark(client, image_path, num_requests, concurrency):
    """
    Запускає бенчмаркінг

    Параметри:
    -----------
    client: клієнт (RestClient або GrpcClient)
    image_path: шлях до файлу зображення
    num_requests: кількість запитів
    concurrency: рівень паралелізму

    Повертає:
    -----------
    словник з результатами
    """
    results = []
    errors = 0

    def send_request():
        response, elapsed, success = client.predict(image_path)
        if not success:
            nonlocal errors
            errors += 1
        return {'response': response, 'elapsed': elapsed, 'success': success}

    start_time = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(send_request) for _ in range(num_requests)]

        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.append(result)

    total_time = time.time() - start_time

    # Обчислення статистики
    latencies = [r['elapsed'] for r in results if r['success']]

    if not latencies:
        print("Всі запити завершилися з помилками")
        return {
            'total_requests': num_requests,
            'successful_requests': 0,
            'failed_requests': num_requests,
            'total_time': total_time,
            'concurrency': concurrency,
            'stats': None
        }

    # Базова статистика
    stats = {
        'min': min(latencies) * 1000,  # мс
        'max': max(latencies) * 1000,  # мс
        'mean': statistics.mean(latencies) * 1000,  # мс
        'median': statistics.median(latencies) * 1000,  # мс
        'p90': np.percentile(latencies, 90) * 1000,  # мс
        'p95': np.percentile(latencies, 95) * 1000,  # мс
        'p99': np.percentile(latencies, 99) * 1000,  # мс
        'std': statistics.stdev(latencies) * 1000 if len(latencies) > 1 else 0,  # мс
        'rps': len(latencies) / total_time  # запитів на секунду
    }

    return {
        'total_requests': num_requests,
        'successful_requests': num_requests - errors,
        'failed_requests': errors,
        'total_time': total_time,
        'concurrency': concurrency,
        'stats': stats,
        'raw_latencies': latencies
    }

def print_benchmark_results(protocol, results):
    """
    Виводить результати бенчмаркінгу

    Параметри:
    -----------
    protocol: назва протоколу (REST або gRPC)
    results: словник з результатами
    """
    print(f"\nРезультати бенчмаркінгу {protocol}:")
    print(f"Загальний час: {results['total_time']:.2f} с")
    print(f"Запитів: {results['total_requests']}")
    print(f"Успішних: {results['successful_requests']} ({100 * results['successful_requests'] / results['total_requests']:.2f}%)")
    print(f"Невдалих: {results['failed_requests']}")
    print(f"Рівень паралелізму: {results['concurrency']}")

    if results['stats']:
        stats = results['stats']
        print("\nСтатистика часу виконання (мс):")
        print(f"  Мін: {stats['min']:.2f}")
        print(f"  Макс: {stats['max']:.2f}")
        print(f"  Середнє: {stats['mean']:.2f}")
        print(f"  Медіана: {stats['median']:.2f}")
        print(f"  P90: {stats['p90']:.2f}")
        print(f"  P95: {stats['p95']:.2f}")
        print(f"  P99: {stats['p99']:.2f}")
        print(f"  Стандартне відхилення: {stats['std']:.2f}")
        print(f"  Запитів на секунду (RPS): {stats['rps']:.2f}")

def compare_and_plot(rest_results, grpc_results, output_file=None):
    """
    Порівнює та візуалізує результати бенчмаркінгу REST та gRPC

    Параметри:
    -----------
    rest_results: словник з результатами REST
    grpc_results: словник з результатами gRPC
    output_file: шлях до вихідного файлу (якщо None, графіки відображаються)
    """
    if not rest_results['stats'] or not grpc_results['stats']:
        print("Недостатньо даних для порівняння")
        return

    # Створення датафрейму для візуалізації
    rest_latencies = [l * 1000 for l in rest_results['raw_latencies']]  # мс
    grpc_latencies = [l * 1000 for l in grpc_results['raw_latencies']]  # мс

    rest_df = pd.DataFrame({'latency': rest_latencies, 'protocol': 'REST'})
    grpc_df = pd.DataFrame({'latency': grpc_latencies, 'protocol': 'gRPC'})
    df = pd.concat([rest_df, grpc_df])

    # Створення графіків
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Порівняння REST та gRPC', fontsize=16)

    # Графік 1: RPS
    protocols = ['REST', 'gRPC']
    rps = [rest_results['stats']['rps'], grpc_results['stats']['rps']]

    axes[0, 0].bar(protocols, rps)
    axes[0, 0].set_title('Запити на секунду (RPS)')
    axes[0, 0].set_ylabel('RPS')
    for i, v in enumerate(rps):
        axes[0, 0].text(i, v, f"{v:.2f}", ha='center', va='bottom')

    # Графік 2: Розподіл затримок
    sns.boxplot(x='protocol', y='latency', data=df, ax=axes[0, 1])
    axes[0, 1].set_title('Розподіл затримок')
    axes[0, 1].set_ylabel('Затримка (мс)')

    # Графік 3: Гістограма затримок
    sns.histplot(data=df, x='latency', hue='protocol', kde=True, ax=axes[1, 0])
    axes[1, 0].set_title('Гістограма затримок')
    axes[1, 0].set_xlabel('Затримка (мс)')
    axes[1, 0].set_ylabel('Кількість запитів')

    # Графік 4: Порівняння метрик
    metrics = ['mean', 'median', 'p90', 'p95', 'p99']
    rest_metrics = [rest_results['stats'][m] for m in metrics]
    grpc_metrics = [grpc_results['stats'][m] for m in metrics]

    x = np.arange(len(metrics))
    width = 0.35

    axes[1, 1].bar(x - width/2, rest_metrics, width, label='REST')
    axes[1, 1].bar(x + width/2, grpc_metrics, width, label='gRPC')

    axes[1, 1].set_title('Порівняння метрик')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(metrics)
    axes[1, 1].set_ylabel('Затримка (мс)')
    axes[1, 1].legend()

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if output_file:
        plt.savefig(output_file)
        print(f"Графіки збережено у {output_file}")
    else:
        plt.show()

def save_results_json(rest_results, grpc_results, output_file):
    """
    Зберігає результати у JSON файл

    Параметри:
    -----------
    rest_results: словник з результатами REST
    grpc_results: словник з результатами gRPC
    output_file: шлях до вихідного файлу
    """
    # Створення копій без raw_latencies для компактності
    rest_copy = rest_results.copy()
    if 'raw_latencies' in rest_copy:
        del rest_copy['raw_latencies']

    grpc_copy = grpc_results.copy() if grpc_results else None
    if grpc_copy and 'raw_latencies' in grpc_copy:
        del grpc_copy['raw_latencies']

    results = {
        'rest': rest_copy,
        'grpc': grpc_copy
    }

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Результати збережено у {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Порівняльний бенчмаркінг REST та gRPC')

    # Основні параметри
    parser.add_argument('--image', type=str, required=True,
                        help='Шлях до тестового зображення')
    parser.add_argument('--rest-url', type=str, default='http://localhost:5000',
                        help='URL для REST API')
    parser.add_argument('--grpc-server', type=str, default='localhost:50051',
                        help='Адреса gRPC сервера')

    # Параметри бенчмаркінгу
    parser.add_argument('--requests', type=int, default=100,
                        help='Кількість запитів')
    parser.add_argument('--concurrency', type=int, default=10,
                        help='Рівень паралелізму')

    # Параметри вихідних даних
    parser.add_argument('--output-json', type=str, default=None,
                        help='Шлях для збереження результатів у JSON')
    parser.add_argument('--output-plot', type=str, default=None,
                        help='Шлях для збереження графіків')

    args = parser.parse_args()

    # Перевірка наявності файлу зображення
    if not os.path.isfile(args.image):
        print(f"Помилка: файл {args.image} не існує")
        return 1

    # REST бенчмаркінг
    print(f"\nЗапуск REST бенчмаркінгу ({args.requests} запитів, паралелізм {args.concurrency})...")
    rest_client = RestClient(args.rest_url)
    rest_results = run_benchmark(rest_client, args.image, args.requests, args.concurrency)
    print_benchmark_results("REST", rest_results)

    # gRPC бенчмаркінг
    grpc_results = None
    if grpc_available:
        try:
            print(f"\nЗапуск gRPC бенчмаркінгу ({args.requests} запитів, паралелізм {args.concurrency})...")
            grpc_client = GrpcClient(args.grpc_server)
            grpc_results = run_benchmark(grpc_client, args.image, args.requests, args.concurrency)
            print_benchmark_results("gRPC", grpc_results)
            grpc_client.close()
        except Exception as e:
            print(f"Помилка при gRPC бенчмаркінгу: {e}")
    else:
        print("\ngRPC бенчмаркінг недоступний (модулі не знайдено)")

    # Порівняння та візуалізація
    if grpc_results and grpc_results['stats'] and rest_results['stats']:
        compare_and_plot(rest_results, grpc_results, args.output_plot)

    # Збереження результатів
    if args.output_json:
        save_results_json(rest_results, grpc_results, args.output_json)

    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
