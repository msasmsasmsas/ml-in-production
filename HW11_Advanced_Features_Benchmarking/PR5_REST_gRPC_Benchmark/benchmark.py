#!/usr/bin/env python
"""
Скрипт для порівняльного бенчмаркінгу REST та gRPC інтерфейсів
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
import logging

# Налаштування логування
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('benchmark')

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
    logger.warning("gRPC модулі не знайдено. gRPC бенчмаркінг буде недоступний.")

class RestClient:
    """
    Клієнт для REST API
    """
    def __init__(self, base_url, timeout=30):
        """
        Ініціалізація клієнта

        Параметри:
        -----------
        base_url: базова URL адреса сервера
        timeout: таймаут для запитів
        """
        self.base_url = base_url
        self.predict_url = f"{base_url}/predict"
        self.health_url = f"{base_url}/health"
        self.timeout = timeout

    def check_health(self):
        """
        Перевірка стану сервера

        Повертає:
        -----------
        (статус, повідомлення)
        """
        try:
            response = requests.get(self.health_url, timeout=self.timeout)
            if response.status_code == 200:
                return True, response.json()
            else:
                return False, f"HTTP помилка: {response.status_code}"
        except Exception as e:
            return False, str(e)

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
                response = requests.post(self.predict_url, files=files, timeout=self.timeout)
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
    def __init__(self, server_address, timeout=30):
        """
        Ініціалізація клієнта

        Параметри:
        -----------
        server_address: адреса gRPC сервера
        timeout: таймаут для запитів
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
        self.timeout = timeout

    def check_health(self):
        """
        Перевірка стану сервера

        Повертає:
        -----------
        (статус, повідомлення)
        """
        try:
            request = inference_pb2.HealthCheckRequest()
            response = self.stub.HealthCheck(request, timeout=self.timeout)

            if response.status == inference_pb2.ServingStatus.SERVING:
                return True, {'status': 'ok', 'metadata': dict(response.metadata)}
            else:
                return False, f"Сервер не готовий: {response.status}"
        except Exception as e:
            return False, str(e)

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
            response = self.stub.Predict(request, timeout=self.timeout)
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

class BenchmarkSuite:
    """
    Набір інструментів для бенчмаркінгу REST та gRPC
    """
    def __init__(self, rest_url=None, grpc_server=None, timeout=30):
        """
        Ініціалізація набору бенчмаркінгу

        Параметри:
        -----------
        rest_url: URL REST API
        grpc_server: адреса gRPC сервера
        timeout: таймаут для запитів
        """
        self.rest_client = None
        self.grpc_client = None
        self.timeout = timeout

        if rest_url:
            self.rest_client = RestClient(rest_url, timeout=timeout)

        if grpc_server and grpc_available:
            try:
                self.grpc_client = GrpcClient(grpc_server, timeout=timeout)
            except Exception as e:
                logger.error(f"Помилка ініціалізації gRPC клієнта: {e}")

    def check_server_health(self):
        """
        Перевірка стану серверів

        Повертає:
        -----------
        (rest_status, grpc_status)
        """
        rest_status = (False, "REST клієнт не ініціалізовано")
        grpc_status = (False, "gRPC клієнт не ініціалізовано")

        if self.rest_client:
            rest_status = self.rest_client.check_health()
            logger.info(f"REST сервер: {'готовий' if rest_status[0] else 'не готовий'} - {rest_status[1]}")

        if self.grpc_client:
            grpc_status = self.grpc_client.check_health()
            logger.info(f"gRPC сервер: {'готовий' if grpc_status[0] else 'не готовий'} - {grpc_status[1]}")

        return rest_status, grpc_status

    def run_benchmark(self, protocol, image_path, num_requests, concurrency):
        """
        Запускає бенчмаркінг для вказаного протоколу

        Параметри:
        -----------
        protocol: протокол ('rest' або 'grpc')
        image_path: шлях до файлу зображення
        num_requests: кількість запитів
        concurrency: рівень паралелізму

        Повертає:
        -----------
        словник з результатами
        """
        if protocol == 'rest' and self.rest_client:
            return self._run_protocol_benchmark(self.rest_client, image_path, num_requests, concurrency)
        elif protocol == 'grpc' and self.grpc_client:
            return self._run_protocol_benchmark(self.grpc_client, image_path, num_requests, concurrency)
        else:
            logger.error(f"Неможливо запустити бенчмаркінг для протоколу {protocol}")
            return None

    def _run_protocol_benchmark(self, client, image_path, num_requests, concurrency):
        """
        Запускає бенчмаркінг для конкретного клієнта

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
        server_times = []

        def send_request():
            response, elapsed, success = client.predict(image_path)

            if not success:
                nonlocal errors
                errors += 1
            else:
                # Збереження часу обробки на сервері, якщо доступно
                if 'processing_time' in response:
                    server_times.append(response['processing_time'] / 1000)  # конвертація з мс у секунди

            return {'response': response, 'elapsed': elapsed, 'success': success}

        logger.info(f"Запуск {num_requests} запитів з паралелізмом {concurrency}...")
        start_time = time.time()

        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [executor.submit(send_request) for _ in range(num_requests)]

            for i, future in enumerate(concurrent.futures.as_completed(futures)):
                result = future.result()
                results.append(result)

                # Логування прогресу
                if (i+1) % max(1, num_requests // 10) == 0 or i+1 == num_requests:
                    logger.info(f"Завершено {i+1}/{num_requests} запитів")

        total_time = time.time() - start_time

        # Обчислення статистики
        latencies = [r['elapsed'] for r in results if r['success']]

        if not latencies:
            logger.error("Всі запити завершилися з помилками")
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

        # Статистика часу сервера, якщо доступна
        if server_times:
            stats['server_time'] = {
                'min': min(server_times) * 1000,  # мс
                'max': max(server_times) * 1000,  # мс
                'mean': statistics.mean(server_times) * 1000,  # мс
                'median': statistics.median(server_times) * 1000,  # мс
                'p90': np.percentile(server_times, 90) * 1000,  # мс
                'p95': np.percentile(server_times, 95) * 1000,  # мс
                'p99': np.percentile(server_times, 99) * 1000  # мс
            }

            # Обчислення мережевої затримки
            network_times = [l - s for l, s in zip(latencies, server_times)]
            stats['network_time'] = {
                'min': min(network_times) * 1000,  # мс
                'max': max(network_times) * 1000,  # мс
                'mean': statistics.mean(network_times) * 1000,  # мс
                'median': statistics.median(network_times) * 1000,  # мс
                'p90': np.percentile(network_times, 90) * 1000,  # мс
                'p95': np.percentile(network_times, 95) * 1000,  # мс
                'p99': np.percentile(network_times, 99) * 1000  # мс
            }

        return {
            'total_requests': num_requests,
            'successful_requests': num_requests - errors,
            'failed_requests': errors,
            'total_time': total_time,
            'concurrency': concurrency,
            'stats': stats,
            'raw_latencies': latencies,
            'raw_server_times': server_times
        }

    def run_comparison(self, image_path, num_requests, concurrency):
        """
        Запускає порівняльний бенчмаркінг REST та gRPC

        Параметри:
        -----------
        image_path: шлях до файлу зображення
        num_requests: кількість запитів
        concurrency: рівень паралелізму

        Повертає:
        -----------
        (rest_results, grpc_results)
        """
        rest_results = None
        grpc_results = None

        # Перевірка стану серверів
        rest_status, grpc_status = self.check_server_health()

        # REST бенчмаркінг
        if self.rest_client and rest_status[0]:
            logger.info("Запуск REST бенчмаркінгу...")
            rest_results = self.run_benchmark('rest', image_path, num_requests, concurrency)
        else:
            logger.warning("REST бенчмаркінг пропущено (сервер не готовий)")

        # gRPC бенчмаркінг
        if self.grpc_client and grpc_status[0]:
            logger.info("Запуск gRPC бенчмаркінгу...")
            grpc_results = self.run_benchmark('grpc', image_path, num_requests, concurrency)
        else:
            logger.warning("gRPC бенчмаркінг пропущено (сервер не готовий)")

        return rest_results, grpc_results

    def close(self):
        """
        Закриття клієнтів
        """
        if self.grpc_client:
            self.grpc_client.close()

def print_benchmark_results(protocol, results):
    """
    Виводить результати бенчмаркінгу

    Параметри:
    -----------
    protocol: назва протоколу (REST або gRPC)
    results: словник з результатами
    """
    if not results:
        logger.warning(f"Результати {protocol} відсутні")
        return

    logger.info(f"\nРезультати бенчмаркінгу {protocol}:")
    logger.info(f"Загальний час: {results['total_time']:.2f} с")
    logger.info(f"Запитів: {results['total_requests']}")
    logger.info(f"Успішних: {results['successful_requests']} ({100 * results['successful_requests'] / results['total_requests']:.2f}%)")
    logger.info(f"Невдалих: {results['failed_requests']}")
    logger.info(f"Рівень паралелізму: {results['concurrency']}")

    if results['stats']:
        stats = results['stats']
        logger.info("\nСтатистика часу виконання (мс):")
        logger.info(f"  Мін: {stats['min']:.2f}")
        logger.info(f"  Макс: {stats['max']:.2f}")
        logger.info(f"  Середнє: {stats['mean']:.2f}")
        logger.info(f"  Медіана: {stats['median']:.2f}")
        logger.info(f"  P90: {stats['p90']:.2f}")
        logger.info(f"  P95: {stats['p95']:.2f}")
        logger.info(f"  P99: {stats['p99']:.2f}")
        logger.info(f"  Стандартне відхилення: {stats['std']:.2f}")
        logger.info(f"  Запитів на секунду (RPS): {stats['rps']:.2f}")

        if 'server_time' in stats:
            logger.info("\nЧас обробки на сервері (мс):")
            logger.info(f"  Мін: {stats['server_time']['min']:.2f}")
            logger.info(f"  Макс: {stats['server_time']['max']:.2f}")
            logger.info(f"  Середнє: {stats['server_time']['mean']:.2f}")
            logger.info(f"  Медіана: {stats['server_time']['median']:.2f}")
            logger.info(f"  P95: {stats['server_time']['p95']:.2f}")

        if 'network_time' in stats:
            logger.info("\nМережева затримка (мс):")
            logger.info(f"  Мін: {stats['network_time']['min']:.2f}")
            logger.info(f"  Макс: {stats['network_time']['max']:.2f}")
            logger.info(f"  Середнє: {stats['network_time']['mean']:.2f}")
            logger.info(f"  Медіана: {stats['network_time']['median']:.2f}")
            logger.info(f"  P95: {stats['network_time']['p95']:.2f}")

def compare_and_plot(rest_results, grpc_results, output_file=None):
    """
    Порівнює та візуалізує результати бенчмаркінгу REST та gRPC

    Параметри:
    -----------
    rest_results: словник з результатами REST
    grpc_results: словник з результатами gRPC
    output_file: шлях до вихідного файлу (якщо None, графіки відображаються)

    Повертає:
    -----------
    True, якщо візуалізація успішна
    """
    if not rest_results or not rest_results['stats'] or \
       not grpc_results or not grpc_results['stats']:
        logger.error("Недостатньо даних для порівняння")
        return False

    # Створення датафрейму для візуалізації
    rest_latencies = [l * 1000 for l in rest_results['raw_latencies']]  # мс
    grpc_latencies = [l * 1000 for l in grpc_results['raw_latencies']]  # мс

    rest_df = pd.DataFrame({'latency': rest_latencies, 'protocol': 'REST'})
    grpc_df = pd.DataFrame({'latency': grpc_latencies, 'protocol': 'gRPC'})
    df = pd.concat([rest_df, grpc_df])

    # Налаштування стилю графіків
    sns.set(style="whitegrid")

    # Створення графіків
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Порівняння REST та gRPC', fontsize=16)

    # Графік 1: RPS
    protocols = ['REST', 'gRPC']
    rps = [rest_results['stats']['rps'], grpc_results['stats']['rps']]

    axes[0, 0].bar(protocols, rps, color=['#3498db', '#2ecc71'])
    axes[0, 0].set_title('Запити на секунду (RPS)')
    axes[0, 0].set_ylabel('RPS')
    for i, v in enumerate(rps):
        axes[0, 0].text(i, v, f"{v:.2f}", ha='center', va='bottom')

    # Графік 2: Розподіл затримок
    sns.boxplot(x='protocol', y='latency', data=df, ax=axes[0, 1], 
                palette={'REST': '#3498db', 'gRPC': '#2ecc71'})
    axes[0, 1].set_title('Розподіл затримок')
    axes[0, 1].set_ylabel('Затримка (мс)')

    # Графік 3: Гістограма затримок
    sns.histplot(data=df, x='latency', hue='protocol', kde=True, ax=axes[1, 0],
                 palette={'REST': '#3498db', 'gRPC': '#2ecc71'})
    axes[1, 0].set_title('Гістограма затримок')
    axes[1, 0].set_xlabel('Затримка (мс)')
    axes[1, 0].set_ylabel('Кількість запитів')

    # Графік 4: Порівняння метрик
    metrics = ['mean', 'median', 'p90', 'p95', 'p99']
    rest_metrics = [rest_results['stats'][m] for m in metrics]
    grpc_metrics = [grpc_results['stats'][m] for m in metrics]

    x = np.arange(len(metrics))
    width = 0.35

    axes[1, 1].bar(x - width/2, rest_metrics, width, label='REST', color='#3498db')
    axes[1, 1].bar(x + width/2, grpc_metrics, width, label='gRPC', color='#2ecc71')

    axes[1, 1].set_title('Порівняння метрик')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(metrics)
    axes[1, 1].set_ylabel('Затримка (мс)')
    axes[1, 1].legend()

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if output_file:
        plt.savefig(output_file)
        logger.info(f"Графіки збережено у {output_file}")
    else:
        plt.show()

    return True

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
    results = {}

    if rest_results:
        rest_copy = rest_results.copy()
        if 'raw_latencies' in rest_copy:
            del rest_copy['raw_latencies']
        if 'raw_server_times' in rest_copy:
            del rest_copy['raw_server_times']
        results['rest'] = rest_copy

    if grpc_results:
        grpc_copy = grpc_results.copy()
        if 'raw_latencies' in grpc_copy:
            del grpc_copy['raw_latencies']
        if 'raw_server_times' in grpc_copy:
            del grpc_copy['raw_server_times']
        results['grpc'] = grpc_copy

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"Результати збережено у {output_file}")

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
    parser.add_argument('--timeout', type=int, default=30,
                        help='Таймаут для запитів (секунди)')

    # Параметри вихідних даних
    parser.add_argument('--output-json', type=str, default=None,
                        help='Шлях для збереження результатів у JSON')
    parser.add_argument('--output-plot', type=str, default=None,
                        help='Шлях для збереження графіків')

    args = parser.parse_args()

    # Перевірка наявності файлу зображення
    if not os.path.isfile(args.image):
        logger.error(f"Помилка: файл {args.image} не існує")
        return 1

    # Ініціалізація набору бенчмаркінгу
    suite = BenchmarkSuite(args.rest_url, args.grpc_server, args.timeout)

    try:
        # Запуск порівняльного бенчмаркінгу
        rest_results, grpc_results = suite.run_comparison(
            args.image, args.requests, args.concurrency
        )

        # Виведення результатів
        print_benchmark_results("REST", rest_results)
        print_benchmark_results("gRPC", grpc_results)

        # Порівняння та візуалізація
        if rest_results and grpc_results:
            compare_and_plot(rest_results, grpc_results, args.output_plot)

        # Збереження результатів
        if args.output_json:
            save_results_json(rest_results, grpc_results, args.output_json)

    finally:
        # Закриття клієнтів
        suite.close()

    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
