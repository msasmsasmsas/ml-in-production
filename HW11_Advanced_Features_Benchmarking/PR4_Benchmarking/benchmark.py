#!/usr/bin/env python
"""
Інструмент для бенчмаркінгу серверів моделей машинного навчання
"""

import os
import time
import json
import argparse
import concurrent.futures
import statistics
import csv
import matplotlib.pyplot as plt
import numpy as np
import requests
import grpc

# Спробуємо імпортувати gRPC модулі, якщо вони є
try:
    import inference_pb2
    import inference_pb2_grpc
    grpc_available = True
except ImportError:
    grpc_available = False
    print("УВАГА: gRPC модулі не знайдено. gRPC бенчмаркінг буде недоступний.")

class BenchmarkResult:
    """
    Клас для зберігання результатів бенчмаркінгу
    """
    def __init__(self, name, protocol):
        """
        Ініціалізація результатів бенчмаркінгу

        Параметри:
        -----------
        name: назва тесту
        protocol: протокол (REST або gRPC)
        """
        self.name = name
        self.protocol = protocol
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.total_time = 0
        self.latencies = []
        self.server_times = []
        self.network_times = []
        self.rps = 0
        self.errors = []
        self.start_time = None
        self.end_time = None
        self.concurrency = 0

    def start(self):
        """
        Початок бенчмаркінгу
        """
        self.start_time = time.time()

    def end(self):
        """
        Завершення бенчмаркінгу
        """
        self.end_time = time.time()
        self.total_time = self.end_time - self.start_time
        if self.total_time > 0:
            self.rps = self.successful_requests / self.total_time

    def add_result(self, success, latency, server_time=None):
        """
        Додавання результату запиту

        Параметри:
        -----------
        success: успішність запиту
        latency: час виконання запиту
        server_time: час обробки на сервері
        """
        self.total_requests += 1

        if success:
            self.successful_requests += 1
            self.latencies.append(latency)

            if server_time is not None:
                self.server_times.append(server_time)
                self.network_times.append(latency - server_time)
        else:
            self.failed_requests += 1

    def add_error(self, error):
        """
        Додавання помилки

        Параметри:
        -----------
        error: текст помилки
        """
        self.errors.append(str(error))

    def get_statistics(self):
        """
        Обчислення статистики

        Повертає:
        -----------
        словник зі статистикою
        """
        stats = {
            'name': self.name,
            'protocol': self.protocol,
            'total_requests': self.total_requests,
            'successful_requests': self.successful_requests,
            'failed_requests': self.failed_requests,
            'success_rate': (self.successful_requests / self.total_requests * 100) if self.total_requests > 0 else 0,
            'total_time': self.total_time,
            'requests_per_second': self.rps,
            'concurrency': self.concurrency
        }

        if self.latencies:
            stats.update({
                'latency': {
                    'min': min(self.latencies) * 1000,  # в мс
                    'max': max(self.latencies) * 1000,  # в мс
                    'mean': statistics.mean(self.latencies) * 1000,  # в мс
                    'median': statistics.median(self.latencies) * 1000,  # в мс
                    'p90': np.percentile(self.latencies, 90) * 1000,  # в мс
                    'p95': np.percentile(self.latencies, 95) * 1000,  # в мс
                    'p99': np.percentile(self.latencies, 99) * 1000   # в мс
                }
            })

        if self.server_times:
            stats.update({
                'server_time': {
                    'min': min(self.server_times) * 1000,  # в мс
                    'max': max(self.server_times) * 1000,  # в мс
                    'mean': statistics.mean(self.server_times) * 1000,  # в мс
                    'median': statistics.median(self.server_times) * 1000,  # в мс
                    'p90': np.percentile(self.server_times, 90) * 1000,  # в мс
                    'p95': np.percentile(self.server_times, 95) * 1000,  # в мс
                    'p99': np.percentile(self.server_times, 99) * 1000   # в мс
                }
            })

        if self.network_times:
            stats.update({
                'network_time': {
                    'min': min(self.network_times) * 1000,  # в мс
                    'max': max(self.network_times) * 1000,  # в мс
                    'mean': statistics.mean(self.network_times) * 1000,  # в мс
                    'median': statistics.median(self.network_times) * 1000,  # в мс
                    'p90': np.percentile(self.network_times, 90) * 1000,  # в мс
                    'p95': np.percentile(self.network_times, 95) * 1000,  # в мс
                    'p99': np.percentile(self.network_times, 99) * 1000   # в мс
                }
            })

        if self.errors:
            stats['errors'] = self.errors[:10]  # обмежуємо кількість помилок
            stats['error_count'] = len(self.errors)

        return stats

class RestClient:
    """
    Клієнт для бенчмаркінгу REST API
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
        self.timeout = timeout

    def predict(self, image_path):
        """
        Відправляє запит на прогнозування

        Параметри:
        -----------
        image_path: шлях до файлу зображення

        Повертає:
        -----------
        кортеж (успіх, час виконання, час сервера)
        """
        try:
            with open(image_path, 'rb') as f:
                files = {'file': (os.path.basename(image_path), f, 'image/jpeg')}

                start_time = time.time()
                response = requests.post(f"{self.base_url}/predict", files=files, timeout=self.timeout)
                elapsed = time.time() - start_time

                if response.status_code == 200:
                    result = response.json()
                    # Спробуємо отримати час обробки на сервері, якщо доступно
                    server_time = result.get('processing_time', None)
                    if server_time is not None:
                        server_time = server_time / 1000  # конвертуємо з мс у секунди
                    return True, elapsed, server_time
                else:
                    return False, elapsed, None
        except Exception as e:
            return False, 0, None

class GrpcClient:
    """
    Клієнт для бенчмаркінгу gRPC API
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
            raise ImportError("gRPC модулі не доступні")

        # Створення каналу з опціями для великих повідомлень
        channel_options = [
            ('grpc.max_send_message_length', 50 * 1024 * 1024),  # 50 MB
            ('grpc.max_receive_message_length', 50 * 1024 * 1024),  # 50 MB
        ]
        self.channel = grpc.insecure_channel(server_address, options=channel_options)
        self.stub = inference_pb2_grpc.InferenceServiceStub(self.channel)
        self.timeout = timeout

    def predict(self, image_path):
        """
        Відправляє запит на прогнозування

        Параметри:
        -----------
        image_path: шлях до файлу зображення

        Повертає:
        -----------
        кортеж (успіх, час виконання, час сервера)
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

            if response.success:
                server_time = response.processing_time / 1000  # конвертуємо з мс у секунди
                return True, elapsed, server_time
            else:
                return False, elapsed, None

        except Exception as e:
            return False, 0, None

    def close(self):
        """
        Закриття з'єднання
        """
        self.channel.close()

def run_benchmark(client, image_path, num_requests, concurrency, result):
    """
    Запускає бенчмаркінг

    Параметри:
    -----------
    client: клієнт (RestClient або GrpcClient)
    image_path: шлях до файлу зображення
    num_requests: кількість запитів
    concurrency: рівень паралелізму
    result: об'єкт BenchmarkResult для збереження результатів

    Повертає:
    -----------
    об'єкт BenchmarkResult з результатами
    """
    result.concurrency = concurrency
    result.start()

    def send_request():
        try:
            success, latency, server_time = client.predict(image_path)
            result.add_result(success, latency, server_time)
            if not success:
                result.add_error("Запит невдалий")
        except Exception as e:
            result.add_result(False, 0)
            result.add_error(str(e))

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(send_request) for _ in range(num_requests)]
        concurrent.futures.wait(futures)

    result.end()
    return result

def print_results(result):
    """
    Виводить результати бенчмаркінгу

    Параметри:
    -----------
    result: об'єкт BenchmarkResult
    """
    stats = result.get_statistics()

    print(f"\nРезультати бенчмаркінгу для {stats['name']} ({stats['protocol']}):\n")
    print(f"Загальний час: {stats['total_time']:.2f} с")
    print(f"Запитів: {stats['total_requests']}")
    print(f"Успішних: {stats['successful_requests']} ({stats['success_rate']:.2f}%)")
    print(f"Невдалих: {stats['failed_requests']}")
    print(f"Запитів на секунду (RPS): {stats['requests_per_second']:.2f}")
    print(f"Рівень паралелізму: {stats['concurrency']}")

    if 'latency' in stats:
        print("\nЧас виконання запиту (мс):")
        print(f"  Мін: {stats['latency']['min']:.2f}")
        print(f"  Макс: {stats['latency']['max']:.2f}")
        print(f"  Середнє: {stats['latency']['mean']:.2f}")
        print(f"  Медіана: {stats['latency']['median']:.2f}")
        print(f"  P90: {stats['latency']['p90']:.2f}")
        print(f"  P95: {stats['latency']['p95']:.2f}")
        print(f"  P99: {stats['latency']['p99']:.2f}")

    if 'server_time' in stats:
        print("\nЧас обробки на сервері (мс):")
        print(f"  Мін: {stats['server_time']['min']:.2f}")
        print(f"  Макс: {stats['server_time']['max']:.2f}")
        print(f"  Середнє: {stats['server_time']['mean']:.2f}")
        print(f"  Медіана: {stats['server_time']['median']:.2f}")
        print(f"  P90: {stats['server_time']['p90']:.2f}")
        print(f"  P95: {stats['server_time']['p95']:.2f}")
        print(f"  P99: {stats['server_time']['p99']:.2f}")

    if 'network_time' in stats:
        print("\nМережева затримка (мс):")
        print(f"  Мін: {stats['network_time']['min']:.2f}")
        print(f"  Макс: {stats['network_time']['max']:.2f}")
        print(f"  Середнє: {stats['network_time']['mean']:.2f}")
        print(f"  Медіана: {stats['network_time']['median']:.2f}")
        print(f"  P90: {stats['network_time']['p90']:.2f}")
        print(f"  P95: {stats['network_time']['p95']:.2f}")
        print(f"  P99: {stats['network_time']['p99']:.2f}")

    if 'errors' in stats and stats['errors']:
        print(f"\nПомилки ({stats['error_count']} всього):")
        for i, error in enumerate(stats['errors']):
            print(f"  {i+1}. {error}")

def save_results_csv(results, output_file):
    """
    Зберігає результати у CSV файл

    Параметри:
    -----------
    results: список об'єктів BenchmarkResult
    output_file: шлях до вихідного файлу
    """
    stats_list = [result.get_statistics() for result in results]

    # Визначення заголовків
    headers = [
        'name', 'protocol', 'concurrency', 'total_requests', 'successful_requests', 
        'failed_requests', 'success_rate', 'total_time', 'requests_per_second',
        'latency_min', 'latency_max', 'latency_mean', 'latency_median', 'latency_p90', 'latency_p95', 'latency_p99',
        'server_time_min', 'server_time_max', 'server_time_mean', 'server_time_median', 'server_time_p90', 'server_time_p95', 'server_time_p99',
        'network_time_min', 'network_time_max', 'network_time_mean', 'network_time_median', 'network_time_p90', 'network_time_p95', 'network_time_p99'
    ]

    rows = []
    for stats in stats_list:
        row = {
            'name': stats['name'],
            'protocol': stats['protocol'],
            'concurrency': stats['concurrency'],
            'total_requests': stats['total_requests'],
            'successful_requests': stats['successful_requests'],
            'failed_requests': stats['failed_requests'],
            'success_rate': stats['success_rate'],
            'total_time': stats['total_time'],
            'requests_per_second': stats['requests_per_second']
        }

        # Додавання метрик затримки
        if 'latency' in stats:
            for key in ['min', 'max', 'mean', 'median', 'p90', 'p95', 'p99']:
                row[f'latency_{key}'] = stats['latency'][key]
        else:
            for key in ['min', 'max', 'mean', 'median', 'p90', 'p95', 'p99']:
                row[f'latency_{key}'] = None

        # Додавання метрик часу сервера
        if 'server_time' in stats:
            for key in ['min', 'max', 'mean', 'median', 'p90', 'p95', 'p99']:
                row[f'server_time_{key}'] = stats['server_time'][key]
        else:
            for key in ['min', 'max', 'mean', 'median', 'p90', 'p95', 'p99']:
                row[f'server_time_{key}'] = None

        # Додавання метрик мережевої затримки
        if 'network_time' in stats:
            for key in ['min', 'max', 'mean', 'median', 'p90', 'p95', 'p99']:
                row[f'network_time_{key}'] = stats['network_time'][key]
        else:
            for key in ['min', 'max', 'mean', 'median', 'p90', 'p95', 'p99']:
                row[f'network_time_{key}'] = None

        rows.append(row)

    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Результати збережено у {output_file}")

def save_results_json(results, output_file):
    """
    Зберігає результати у JSON файл

    Параметри:
    -----------
    results: список об'єктів BenchmarkResult
    output_file: шлях до вихідного файлу
    """
    stats_list = [result.get_statistics() for result in results]

    with open(output_file, 'w') as f:
        json.dump(stats_list, f, indent=2)

    print(f"Результати збережено у {output_file}")

def plot_results(results, output_file=None):
    """
    Створює візуалізацію результатів

    Параметри:
    -----------
    results: список об'єктів BenchmarkResult
    output_file: шлях до вихідного файлу (якщо None, графіки відображаються)
    """
    stats_list = [result.get_statistics() for result in results]

    # Створення графіків
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Результати бенчмаркінгу', fontsize=16)

    # Графік 1: RPS
    names = [f"{stats['name']}\n({stats['protocol']})" for stats in stats_list]
    rps = [stats['requests_per_second'] for stats in stats_list]

    axes[0, 0].bar(names, rps)
    axes[0, 0].set_title('Запити на секунду (RPS)')
    axes[0, 0].set_ylabel('RPS')
    axes[0, 0].grid(axis='y', linestyle='--', alpha=0.7)

    # Графік 2: Latency
    latencies = []
    for stats in stats_list:
        if 'latency' in stats:
            latencies.append([
                stats['latency']['min'],
                stats['latency']['mean'],
                stats['latency']['p95'],
                stats['latency']['max']
            ])
        else:
            latencies.append([0, 0, 0, 0])

    x = np.arange(len(names))
    width = 0.2

    axes[0, 1].bar(x - width*1.5, [l[0] for l in latencies], width, label='Min')
    axes[0, 1].bar(x - width/2, [l[1] for l in latencies], width, label='Mean')
    axes[0, 1].bar(x + width/2, [l[2] for l in latencies], width, label='P95')
    axes[0, 1].bar(x + width*1.5, [l[3] for l in latencies], width, label='Max')

    axes[0, 1].set_title('Час виконання запиту (мс)')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(names)
    axes[0, 1].set_ylabel('Час (мс)')
    axes[0, 1].legend()
    axes[0, 1].grid(axis='y', linestyle='--', alpha=0.7)

    # Графік 3: Порівняння часу сервера і мережі
    server_times = []
    network_times = []

    for stats in stats_list:
        if 'server_time' in stats and 'network_time' in stats:
            server_times.append(stats['server_time']['mean'])
            network_times.append(stats['network_time']['mean'])
        else:
            server_times.append(0)
            network_times.append(0)

    axes[1, 0].bar(names, server_times, label='Час сервера')
    axes[1, 0].bar(names, network_times, bottom=server_times, label='Мережева затримка')

    axes[1, 0].set_title('Розподіл часу виконання запиту (мс)')
    axes[1, 0].set_ylabel('Час (мс)')
    axes[1, 0].legend()
    axes[1, 0].grid(axis='y', linestyle='--', alpha=0.7)

    # Графік 4: Успішність запитів
    success_rates = [stats['success_rate'] for stats in stats_list]

    axes[1, 1].bar(names, success_rates)
    axes[1, 1].set_title('Відсоток успішних запитів (%)')
    axes[1, 1].set_ylabel('Відсоток (%)')
    axes[1, 1].set_ylim(0, 100)
    axes[1, 1].grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if output_file:
        plt.savefig(output_file)
        print(f"Графіки збережено у {output_file}")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='Інструмент для бенчмаркінгу серверів моделей')

    # Основні параметри
    parser.add_argument('--mode', type=str, choices=['rest', 'grpc', 'both'], default='both',
                        help='Режим бенчмаркінгу: REST, gRPC або обидва')
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
    parser.add_argument('--output-csv', type=str, default=None,
                        help='Шлях для збереження результатів у CSV')
    parser.add_argument('--output-json', type=str, default=None,
                        help='Шлях для збереження результатів у JSON')
    parser.add_argument('--output-plot', type=str, default=None,
                        help='Шлях для збереження графіків')

    # Параметри для порівняння різних конфігурацій
    parser.add_argument('--concurrency-range', type=str, default=None,
                        help='Діапазон рівнів паралелізму у форматі "start,end,step"')

    args = parser.parse_args()

    # Перевірка наявності файлу зображення
    if not os.path.isfile(args.image):
        print(f"Помилка: файл {args.image} не існує")
        return 1

    # Підготовка списку рівнів паралелізму для тестування
    concurrency_levels = [args.concurrency]
    if args.concurrency_range:
        try:
            start, end, step = map(int, args.concurrency_range.split(','))
            concurrency_levels = list(range(start, end + 1, step))
        except ValueError:
            print(f"Помилка: невірний формат діапазону паралелізму: {args.concurrency_range}")
            return 1

    # Запуск бенчмаркінгу
    results = []

    # REST API бенчмаркінг
    if args.mode in ['rest', 'both']:
        rest_client = RestClient(args.rest_url, timeout=args.timeout)

        for concurrency in concurrency_levels:
            print(f"\nЗапуск REST бенчмаркінгу з {args.requests} запитами та рівнем паралелізму {concurrency}...")
            result = BenchmarkResult(f"REST-C{concurrency}", "REST")
            run_benchmark(rest_client, args.image, args.requests, concurrency, result)
            print_results(result)
            results.append(result)

    # gRPC бенчмаркінг
    if args.mode in ['grpc', 'both'] and grpc_available:
        try:
            grpc_client = GrpcClient(args.grpc_server, timeout=args.timeout)

            for concurrency in concurrency_levels:
                print(f"\nЗапуск gRPC бенчмаркінгу з {args.requests} запитами та рівнем паралелізму {concurrency}...")
                result = BenchmarkResult(f"gRPC-C{concurrency}", "gRPC")
                run_benchmark(grpc_client, args.image, args.requests, concurrency, result)
                print_results(result)
                results.append(result)

            grpc_client.close()
        except Exception as e:
            print(f"Помилка при gRPC бенчмаркінгу: {e}")

    # Збереження результатів
    if args.output_csv:
        save_results_csv(results, args.output_csv)

    if args.output_json:
        save_results_json(results, args.output_json)

    if args.output_plot or len(results) > 1:
        plot_results(results, args.output_plot)

    return 0

if __name__ == "__main__":
    sys.exit(main())
