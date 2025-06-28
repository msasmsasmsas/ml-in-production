import os
import time
import argparse
import sys
from pathlib import Path
import grpc
import concurrent.futures

# Імпортуємо згенеровані gRPC модулі
import inference_pb2
import inference_pb2_grpc

class InferenceClient:
    """
    Клієнт для gRPC інференсу моделей машинного навчання
    """
    def __init__(self, server_address):
        """
        Ініціалізація клієнта

        Параметри:
        -----------
        server_address: адреса gRPC сервера
        """
        # Створення каналу з опціями для великих повідомлень
        channel_options = [
            ('grpc.max_send_message_length', 50 * 1024 * 1024),  # 50 MB
            ('grpc.max_receive_message_length', 50 * 1024 * 1024),  # 50 MB
        ]
        self.channel = grpc.insecure_channel(server_address, options=channel_options)
        self.stub = inference_pb2_grpc.InferenceServiceStub(self.channel)

    def health_check(self):
        """
        Перевірка стану сервера

        Повертає:
        -----------
        об'єкт HealthCheckResponse
        """
        request = inference_pb2.HealthCheckRequest()
        return self.stub.HealthCheck(request)

    def predict(self, image_path):
        """
        Відправляє запит на прогнозування зображення

        Параметри:
        -----------
        image_path: шлях до файлу зображення

        Повертає:
        -----------
        об'єкт PredictResponse та час виконання запиту
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

            return response, elapsed

        except Exception as e:
            print(f"Помилка при виконанні запиту: {e}")
            return None, 0

    def predict_stream(self, image_paths):
        """
        Відправляє потоковий запит на прогнозування кількох зображень

        Параметри:
        -----------
        image_paths: список шляхів до файлів зображень

        Повертає:
        -----------
        генератор пар (відповідь, час виконання)
        """
        def request_generator():
            for path in image_paths:
                try:
                    with open(path, 'rb') as f:
                        image_data = f.read()

                    request = inference_pb2.PredictRequest(
                        data=image_data,
                        content_type='image/jpeg',
                        parameters={'image_path': os.path.basename(path)}
                    )

                    yield request

                except Exception as e:
                    print(f"Помилка при підготовці запиту для {path}: {e}")

        try:
            start_time = time.time()
            responses = self.stub.PredictStream(request_generator())

            for response in responses:
                current_time = time.time()
                elapsed = current_time - start_time
                start_time = current_time

                yield response, elapsed

        except Exception as e:
            print(f"Помилка при виконанні потокового запиту: {e}")

    def close(self):
        """
        Закриття з'єднання
        """
        self.channel.close()

def format_prediction(prediction):
    """
    Форматування прогнозу для виведення

    Параметри:
    -----------
    prediction: об'єкт ClassPrediction

    Повертає:
    -----------
    рядок з форматованим прогнозом
    """
    return f"{prediction.class_name} ({prediction.class_id}): {prediction.score:.4f}"

def run_single_request(client, image_path):
    """
    Виконує один запит і виводить результати

    Параметри:
    -----------
    client: екземпляр InferenceClient
    image_path: шлях до файлу зображення
    """
    print(f"Відправлення запиту для {image_path}")

    response, elapsed = client.predict(image_path)

    if response is None:
        print("Не отримано відповіді від сервера")
        return

    print(f"\nРезультати прогнозування:")
    print(f"ID запиту: {response.request_id}")
    print(f"Статус: {'Успішно' if response.success else 'Помилка'}")

    if not response.success:
        print(f"Помилка: {response.error}")
        return

    print(f"Час обробки на сервері: {response.processing_time:.2f} мс")
    print(f"Загальний час запиту: {elapsed*1000:.2f} мс")
    print(f"Мережева затримка: {(elapsed*1000 - response.processing_time):.2f} мс")

    print("\nТоп-5 прогнозів:")
    for i, prediction in enumerate(response.predictions):
        print(f"{i+1}. {format_prediction(prediction)}")

    if response.metadata:
        print("\nМетадані:")
        for key, value in response.metadata.items():
            print(f"{key}: {value}")

def run_benchmark(client, image_path, num_requests, concurrency):
    """
    Виконує benchmark з використанням паралельних запитів

    Параметри:
    -----------
    client: екземпляр InferenceClient
    image_path: шлях до файлу зображення
    num_requests: кількість запитів
    concurrency: кількість паралельних запитів
    """
    print(f"Запуск benchmark: {num_requests} запитів з рівнем паралелізму {concurrency}")

    results = []
    errors = 0

    def send_request(image_path):
        response, elapsed = client.predict(image_path)
        if response is None or not response.success:
            return {'success': False, 'elapsed': elapsed, 'server_time': 0}
        return {
            'success': True, 
            'elapsed': elapsed, 
            'server_time': response.processing_time / 1000  # конвертація з мс в секунди
        }

    start_time = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(send_request, image_path) for _ in range(num_requests)]

        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.append(result)
            if not result['success']:
                errors += 1

    total_time = time.time() - start_time

    # Обчислення статистики
    if results:
        client_times = [r['elapsed'] for r in results if r['success']]
        server_times = [r['server_time'] for r in results if r['success']]

        if client_times:
            avg_client_time = sum(client_times) / len(client_times)
            min_client_time = min(client_times)
            max_client_time = max(client_times)
            p95_client_time = sorted(client_times)[int(len(client_times) * 0.95)]
        else:
            avg_client_time = min_client_time = max_client_time = p95_client_time = 0

        if server_times:
            avg_server_time = sum(server_times) / len(server_times)
            min_server_time = min(server_times)
            max_server_time = max(server_times)
            p95_server_time = sorted(server_times)[int(len(server_times) * 0.95)]
        else:
            avg_server_time = min_server_time = max_server_time = p95_server_time = 0

        rps = num_requests / total_time

        print(f"\nРезультати benchmark:")
        print(f"Загальний час: {total_time:.2f} с")
        print(f"Успішних запитів: {num_requests - errors} з {num_requests} ({100 * (num_requests - errors) / num_requests:.2f}%)")
        print(f"RPS (запитів на секунду): {rps:.2f}")
        print(f"\nЧас виконання запиту (клієнт):")
        print(f"  Середній: {avg_client_time * 1000:.2f} мс")
        print(f"  Мінімальний: {min_client_time * 1000:.2f} мс")
        print(f"  Максимальний: {max_client_time * 1000:.2f} мс")
        print(f"  P95: {p95_client_time * 1000:.2f} мс")
        print(f"\nЧас обробки (сервер):")
        print(f"  Середній: {avg_server_time * 1000:.2f} мс")
        print(f"  Мінімальний: {min_server_time * 1000:.2f} мс")
        print(f"  Максимальний: {max_server_time * 1000:.2f} мс")
        print(f"  P95: {p95_server_time * 1000:.2f} мс")
        print(f"\nМережева затримка (туди-назад):")
        print(f"  Середня: {(avg_client_time - avg_server_time) * 1000:.2f} мс")

        return {
            'total_time': total_time,
            'successful_requests': num_requests - errors,
            'total_requests': num_requests,
            'rps': rps,
            'avg_client_time_ms': avg_client_time * 1000,
            'avg_server_time_ms': avg_server_time * 1000,
            'avg_network_time_ms': (avg_client_time - avg_server_time) * 1000
        }
    else:
        print("Не отримано успішних результатів")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='gRPC клієнт для інференсу моделей')
    parser.add_argument('--server', type=str, default='localhost:50051', help='Адреса gRPC сервера')
    parser.add_argument('--image', type=str, default='test_image.jpg', help='Шлях до тестового зображення')
    parser.add_argument('--mode', type=str, choices=['single', 'benchmark'], default='single', 
                        help='Режим роботи: одиночний запит (single) або тестування продуктивності (benchmark)')
    parser.add_argument('--requests', type=int, default=100, help='Кількість запитів для режиму benchmark')
    parser.add_argument('--concurrency', type=int, default=10, help='Рівень паралелізму для режиму benchmark')

    args = parser.parse_args()

    if not Path(args.image).exists():
        print(f"Помилка: файл {args.image} не існує")
        sys.exit(1)

    client = InferenceClient(args.server)

    try:
        # Перевірка здоров'я сервера
        health_response = client.health_check()
        print(f"Статус сервера: {health_response.status}")

        if health_response.status != inference_pb2.ServingStatus.SERVING:
            print("Сервер не готовий до роботи")
            sys.exit(1)

        print("Сервер готовий до роботи")
        if health_response.metadata:
            print("Інформація про сервер:")
            for key, value in health_response.metadata.items():
                print(f"{key}: {value}")

        # Виконання запитів відповідно до режиму
        if args.mode == 'single':
            run_single_request(client, args.image)
        else:  # benchmark
            run_benchmark(client, args.image, args.requests, args.concurrency)

    except Exception as e:
        print(f"Помилка: {e}")
        sys.exit(1)
    finally:
        client.close()
