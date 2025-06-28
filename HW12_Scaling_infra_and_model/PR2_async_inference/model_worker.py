#!/usr/bin/env python

'''
Воркер для асинхронного інференсу моделей
'''

import os
import time
import json
import logging
import binascii
import threading
import argparse
from typing import Dict, List, Any, Optional, Union

import torch
import numpy as np
from PIL import Image
import io

# Kafka інтеграція
from confluent_kafka import Producer, Consumer, KafkaError, KafkaException

# Налаштування логування
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('model_worker')

class ModelInferenceWorker:
    '''
    Воркер для асинхронного інференсу моделей
    '''

    def __init__(self, 
                bootstrap_servers: str,
                request_topic: str,
                response_topic: str,
                consumer_group: str,
                model_path: str,
                num_workers: int = 2):
        '''
        Ініціалізація воркера

        Параметри:
        -----------
        bootstrap_servers: список Kafka серверів
        request_topic: тема для запитів інференсу
        response_topic: тема для відповідей інференсу
        consumer_group: група споживачів
        model_path: шлях до файлу моделі
        num_workers: кількість воркерів для обробки запитів
        '''
        self.bootstrap_servers = bootstrap_servers
        self.request_topic = request_topic
        self.response_topic = response_topic
        self.consumer_group = consumer_group
        self.model_path = model_path
        self.num_workers = num_workers

        # Налаштування продюсера Kafka
        self.producer_config = {
            'bootstrap.servers': bootstrap_servers,
            'client.id': f'model-worker-producer',
            'acks': 'all',
            'retries': 3,
            'retry.backoff.ms': 100
        }

        # Налаштування споживача Kafka
        self.consumer_config = {
            'bootstrap.servers': bootstrap_servers,
            'group.id': consumer_group,
            'auto.offset.reset': 'earliest',
            'enable.auto.commit': True,
            'auto.commit.interval.ms': 1000,
            'max.poll.interval.ms': 300000
        }

        self.producer = None
        self.consumers = []
        self.worker_threads = []
        self.running = False

        # Завантаження моделі
        self.model = self._load_model(model_path)

        # Обробники моделей за назвами
        self.model_handlers = {
            'default': self._default_inference,
            'image_classification': self._image_classification_inference,
            'text_classification': self._text_classification_inference
        }

    def _load_model(self, model_path):
        '''
        Завантаження моделі

        Параметри:
        -----------
        model_path: шлях до файлу моделі

        Повертає:
        -----------
        завантажена модель
        '''
        try:
            logger.info(f"Завантаження моделі з {model_path}")

            # Визначення типу моделі за розширенням
            if model_path.endswith('.pt') or model_path.endswith('.pth'):
                # PyTorch модель
                if torch.cuda.is_available():
                    logger.info("Використання GPU для інференсу")
                    device = torch.device("cuda")
                else:
                    logger.info("Використання CPU для інференсу")
                    device = torch.device("cpu")

                # Завантаження моделі PyTorch
                model = torch.load(model_path, map_location=device)
                model.eval()  # Перемикання в режим інференсу

                return model

            elif model_path.endswith('.onnx'):
                # ONNX модель
                import onnxruntime as ort

                # Налаштування сесії ONNX Runtime
                sess_options = ort.SessionOptions()
                sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

                # Вибір провайдера
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider']

                # Створення сесії
                session = ort.InferenceSession(model_path, sess_options, providers=providers)

                return session

            else:
                # Заглушка для тестування
                logger.warning(f"Формат моделі не розпізнано, використання заглушки для тестування")
                return DummyModel()

        except Exception as e:
            logger.error(f"Помилка завантаження моделі: {e}")
            # Заглушка для тестування
            return DummyModel()

    def start(self):
        '''
        Запуск воркера
        '''
        if self.running:
            logger.warning("Воркер вже запущено")
            return

        self.running = True

        # Створення продюсера
        self.producer = Producer(self.producer_config)

        # Запуск воркерів
        for i in range(self.num_workers):
            self._start_worker(i)

        logger.info(f"Запущено {self.num_workers} воркерів для обробки запитів")

    def stop(self):
        '''
        Зупинка воркера
        '''
        if not self.running:
            return

        self.running = False

        # Зупинка всіх воркерів
        for consumer in self.consumers:
            consumer.close()

        # Очікування завершення всіх потоків
        for thread in self.worker_threads:
            thread.join(timeout=5.0)

        # Закриття продюсера
        if self.producer:
            self.producer.flush()

        self.consumers = []
        self.worker_threads = []

        logger.info("Воркер зупинено")

    def _start_worker(self, worker_id):
        '''
        Запуск воркера для обробки запитів

        Параметри:
        -----------
        worker_id: ідентифікатор воркера
        '''
        # Створення споживача для воркера
        worker_consumer = Consumer({
            **self.consumer_config,
            'group.id': f"{self.consumer_group}-{worker_id}"
        })
        worker_consumer.subscribe([self.request_topic])

        self.consumers.append(worker_consumer)

        # Запуск воркера в окремому потоці
        worker_thread = threading.Thread(
            target=self._worker_loop,
            args=(worker_id, worker_consumer),
            daemon=True
        )
        worker_thread.start()
        self.worker_threads.append(worker_thread)

        logger.info(f"Запущено воркер {worker_id}")

    def _worker_loop(self, worker_id, consumer):
        '''
        Основний цикл воркера

        Параметри:
        -----------
        worker_id: ідентифікатор воркера
        consumer: споживач Kafka
        '''
        try:
            while self.running:
                msg = consumer.poll(1.0)

                if msg is None:
                    continue

                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        # Кінець партиції
                        continue
                    else:
                        logger.error(f"Помилка споживача: {msg.error()}")
                        continue

                # Обробка повідомлення
                try:
                    self._process_request(msg.key(), msg.value(), worker_id)
                except Exception as e:
                    logger.error(f"Помилка обробки запиту воркером {worker_id}: {e}")
        except Exception as e:
            logger.error(f"Критична помилка в воркері {worker_id}: {e}")
        finally:
            logger.info(f"Воркер {worker_id} завершив роботу")

    def _process_request(self, key, value, worker_id):
        '''
        Обробка запиту інференсу

        Параметри:
        -----------
        key: ключ повідомлення (request_id)
        value: значення повідомлення (дані запиту)
        worker_id: ідентифікатор воркера
        '''
        start_time = time.time()

        try:
            # Декодування ключа та повідомлення
            request_id = key.decode('utf-8')
            request = json.loads(value.decode('utf-8'))

            model_name = request.get('model_name', 'default')
            data = request.get('data')
            timestamp = request.get('timestamp', 0)

            # Перевірка часу очікування в черзі
            queue_time = time.time() - timestamp
            logger.debug(f"Запит {request_id} очікував у черзі {queue_time:.4f} секунд")

            # Вибір обробника на основі назви моделі
            handler = self.model_handlers.get(model_name, self.model_handlers['default'])

            # Виклик обробника
            result = handler(data)

            # Підготовка відповіді
            response = {
                'request_id': request_id,
                'success': True,
                'timestamp': time.time(),
                'queue_time': queue_time,
                'processing_time': time.time() - start_time,
                'result': result
            }

        except Exception as e:
            logger.error(f"Помилка обробки запиту {key}: {e}")
            response = {
                'request_id': key.decode('utf-8') if isinstance(key, bytes) else str(key),
                'success': False,
                'timestamp': time.time(),
                'error': str(e)
            }

        # Відправка відповіді
        try:
            self.producer.produce(
                self.response_topic,
                key=key,
                value=json.dumps(response).encode('utf-8'),
                callback=self._delivery_callback
            )
            self.producer.poll(0)  # Тригер доставки повідомлень
        except Exception as e:
            logger.error(f"Помилка відправки відповіді: {e}")

    def _delivery_callback(self, err, msg):
        '''
        Callback для підтвердження доставки повідомлення
        '''
        if err:
            logger.error(f"Помилка доставки повідомлення: {err}")
        else:
            logger.debug(f"Повідомлення доставлено в {msg.topic()} [{msg.partition()}] в оффсет {msg.offset()}")

    def _default_inference(self, data):
        '''
        Стандартний обробник інференсу

        Параметри:
        -----------
        data: дані для інференсу

        Повертає:
        -----------
        результат інференсу
        '''
        logger.info(f"Виконання стандартного інференсу")

        # Перевірка типу даних
        if isinstance(data, dict) and 'content_type' in data and 'data' in data:
            # Спроба обробити на основі content_type
            content_type = data['content_type']

            if 'image' in content_type:
                return self._image_classification_inference(data)
            elif 'text' in content_type:
                return self._text_classification_inference(data)

        # Використання моделі напряму
        try:
            if hasattr(self.model, 'predict'):
                return self.model.predict(data).tolist()
            elif hasattr(self.model, 'forward'):
                with torch.no_grad():
                    input_tensor = torch.tensor(data).float()
                    output = self.model(input_tensor)
                    return output.cpu().numpy().tolist()
            else:
                # Повертаємо дані для заглушки
                return {"prediction": "default", "score": 0.9, "processing_time": time.time() - start_time}
        except Exception as e:
            logger.error(f"Помилка стандартного інференсу: {e}")
            raise

    def _image_classification_inference(self, data):
        '''
        Обробник інференсу для класифікації зображень

        Параметри:
        -----------
        data: дані для інференсу

        Повертає:
        -----------
        результат інференсу
        '''
        logger.info(f"Виконання інференсу класифікації зображень")
        start_time = time.time()

        try:
            # Обробка вхідних даних
            if isinstance(data, dict) and 'data' in data:
                # Конвертація з hex у байти
                image_bytes = binascii.unhexlify(data['data'])
                image = Image.open(io.BytesIO(image_bytes))
            elif isinstance(data, str):
                # Спроба завантажити з base64
                import base64
                image_bytes = base64.b64decode(data)
                image = Image.open(io.BytesIO(image_bytes))
            else:
                raise ValueError("Непідтримуваний формат даних для класифікації зображень")

            # Попередня обробка зображення
            from torchvision import transforms

            preprocess = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

            image_tensor = preprocess(image).unsqueeze(0)

            # Перенесення на потрібний пристрій
            device = next(self.model.parameters()).device
            image_tensor = image_tensor.to(device)

            # Виконання інференсу
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]

                # Отримання топ-5 класів
                top5_prob, top5_catid = torch.topk(probabilities, 5)

                # Конвертація результатів
                result = {
                    "predictions": [
                        {"class_id": int(idx), "score": float(prob)} 
                        for idx, prob in zip(top5_catid.cpu().numpy(), top5_prob.cpu().numpy())
                    ],
                    "processing_time": time.time() - start_time
                }

                return result

        except Exception as e:
            logger.error(f"Помилка інференсу класифікації зображень: {e}")
            raise

    def _text_classification_inference(self, data):
        '''
        Обробник інференсу для класифікації тексту

        Параметри:
        -----------
        data: дані для інференсу

        Повертає:
        -----------
        результат інференсу
        '''
        logger.info(f"Виконання інференсу класифікації тексту")
        start_time = time.time()

        try:
            # Обробка вхідних даних
            if isinstance(data, dict) and 'data' in data:
                # Текст може бути у форматі UTF-8 hex
                if isinstance(data['data'], str):
                    try:
                        text = binascii.unhexlify(data['data']).decode('utf-8')
                    except:
                        text = data['data']  # Вже текст
                else:
                    text = str(data['data'])
            elif isinstance(data, str):
                text = data
            else:
                raise ValueError("Непідтримуваний формат даних для класифікації тексту")

            # Заглушка для тестування
            result = {
                "predictions": [
                    {"label": "positive", "score": 0.8},
                    {"label": "negative", "score": 0.2}
                ],
                "processing_time": time.time() - start_time
            }

            return result

        except Exception as e:
            logger.error(f"Помилка інференсу класифікації тексту: {e}")
            raise

class DummyModel:
    '''
    Заглушка моделі для тестування
    '''
    def __init__(self):
        self.device = "cpu"

    def __call__(self, input_data):
        # Імітація обробки для тестування
        time.sleep(0.1)  # Імітація затримки інференсу

        if isinstance(input_data, torch.Tensor):
            # Для класифікації зображень
            batch_size = input_data.shape[0]
            return torch.rand(batch_size, 1000)  # Імітація виходу класифікатора з 1000 класами
        else:
            # Для текстової класифікації або іншого типу даних
            return {"prediction": "dummy", "score": 0.9}

    def predict(self, data):
        # Для scikit-learn інтерфейсу
        return np.random.rand(len(data) if hasattr(data, '__len__') else 1)

    def parameters(self):
        # Імітація параметрів моделі
        yield torch.nn.Parameter(torch.randn(1))

def main():
    parser = argparse.ArgumentParser(description="Воркер для асинхронного інференсу моделей")
    parser.add_argument("--bootstrap-servers", type=str, default=os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"),
                        help="Список Kafka серверів")
    parser.add_argument("--request-topic", type=str, default=os.environ.get("KAFKA_REQUEST_TOPIC", "model-inference-requests"),
                        help="Тема для запитів інференсу")
    parser.add_argument("--response-topic", type=str, default=os.environ.get("KAFKA_RESPONSE_TOPIC", "model-inference-responses"),
                        help="Тема для відповідей інференсу")
    parser.add_argument("--consumer-group", type=str, default=os.environ.get("KAFKA_CONSUMER_GROUP", "model-inference-worker"),
                        help="Група споживачів")
    parser.add_argument("--model-path", type=str, default=os.environ.get("MODEL_PATH", "models/model.pt"),
                        help="Шлях до файлу моделі")
    parser.add_argument("--num-workers", type=int, default=int(os.environ.get("NUM_WORKERS", "2")),
                        help="Кількість воркерів для обробки запитів")

    args = parser.parse_args()

    worker = ModelInferenceWorker(
        bootstrap_servers=args.bootstrap_servers,
        request_topic=args.request_topic,
        response_topic=args.response_topic,
        consumer_group=args.consumer_group,
        model_path=args.model_path,
        num_workers=args.num_workers
    )

    worker.start()

    try:
        # Очікування на сигнал завершення
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Отримано сигнал завершення")
    finally:
        worker.stop()

if __name__ == "__main__":
    main()
