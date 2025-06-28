#!/usr/bin/env python

'''
Сервіс асинхронного інференсу з використанням Apache Kafka
'''

import os
import json
import time
import uuid
import threading
import logging
from typing import Dict, List, Any, Optional, Callable

# Kafka клієнт
from confluent_kafka import Producer, Consumer, KafkaError, KafkaException

# Налаштування логування
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('kafka_queue')

class KafkaQueueService:
    '''
    Сервіс асинхронного інференсу з використанням Apache Kafka
    '''

    def __init__(self, 
                bootstrap_servers: str,
                request_topic: str,
                response_topic: str,
                consumer_group: str,
                num_workers: int = 2):
        '''
        Ініціалізація сервісу Kafka черги

        Параметри:
        -----------
        bootstrap_servers: список Kafka серверів
        request_topic: тема для запитів інференсу
        response_topic: тема для відповідей інференсу
        consumer_group: група споживачів
        num_workers: кількість воркерів для обробки запитів
        '''
        self.bootstrap_servers = bootstrap_servers
        self.request_topic = request_topic
        self.response_topic = response_topic
        self.consumer_group = consumer_group
        self.num_workers = num_workers

        # Налаштування продюсера Kafka
        self.producer_config = {
            'bootstrap.servers': bootstrap_servers,
            'client.id': f'model-inference-producer-{uuid.uuid4()}',
            'acks': 'all',  # Очікувати підтвердження від всіх реплік
            'retries': 3,    # Кількість повторних спроб
            'retry.backoff.ms': 100  # Затримка між спробами
        }

        # Налаштування споживача Kafka
        self.consumer_config = {
            'bootstrap.servers': bootstrap_servers,
            'group.id': consumer_group,
            'auto.offset.reset': 'earliest',  # Починати з найстаріших повідомлень
            'enable.auto.commit': True,       # Автоматичний коміт офсетів
            'auto.commit.interval.ms': 1000,  # Інтервал коміту офсетів
            'max.poll.interval.ms': 300000    # Максимальний інтервал між опитуваннями
        }

        self.producer = None
        self.consumers = []
        self.worker_threads = []
        self.running = False

        # Колбеки для обробки запитів
        self.request_handlers = {}

        # Словник для callback-ів очікування відповідей
        self.response_callbacks = {}
        self.response_lock = threading.Lock()

    def start(self):
        '''
        Запуск сервісу Kafka черги
        '''
        if self.running:
            logger.warning("Сервіс вже запущено")
            return

        self.running = True

        # Створення продюсера
        self.producer = Producer(self.producer_config)

        # Запуск обробника відповідей
        self._start_response_consumer()

        # Запуск воркерів
        for i in range(self.num_workers):
            self._start_worker(i)

        logger.info(f"Запущено сервіс Kafka черги з {self.num_workers} воркерами")

    def stop(self):
        '''
        Зупинка сервісу Kafka черги
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
        self.response_callbacks = {}

        logger.info("Сервіс Kafka черги зупинено")

    def register_handler(self, model_name: str, handler_func: Callable):
        '''
        Реєстрація обробника для моделі

        Параметри:
        -----------
        model_name: назва моделі
        handler_func: функція-обробник запитів інференсу
        '''
        self.request_handlers[model_name] = handler_func
        logger.info(f"Зареєстровано обробник для моделі '{model_name}'")

    def submit_inference_request(self, model_name: str, data: Any, callback: Optional[Callable] = None, timeout: int = 30):
        '''
        Відправка запиту на асинхронний інференс

        Параметри:
        -----------
        model_name: назва моделі
        data: дані для інференсу
        callback: функція, яка буде викликана з результатом
        timeout: таймаут очікування відповіді в секундах

        Повертає:
        -----------
        request_id: ідентифікатор запиту
        '''
        request_id = str(uuid.uuid4())

        # Створення повідомлення
        message = {
            'request_id': request_id,
            'model_name': model_name,
            'timestamp': time.time(),
            'data': data
        }

        # Серіалізація повідомлення
        try:
            payload = json.dumps(message).encode('utf-8')
        except (TypeError, ValueError) as e:
            logger.error(f"Помилка серіалізації запиту: {e}")
            if callback:
                callback({'error': f"Помилка серіалізації запиту: {e}", 'request_id': request_id})
            return request_id

        # Реєстрація callback'у для очікування відповіді
        if callback:
            with self.response_lock:
                self.response_callbacks[request_id] = {
                    'callback': callback,
                    'expires': time.time() + timeout
                }

        # Відправка повідомлення в Kafka
        try:
            self.producer.produce(
                self.request_topic,
                key=request_id.encode('utf-8'),
                value=payload,
                callback=self._delivery_callback
            )
            self.producer.poll(0)  # Тригер доставки повідомлень
        except Exception as e:
            logger.error(f"Помилка відправки запиту в Kafka: {e}")
            if callback:
                with self.response_lock:
                    if request_id in self.response_callbacks:
                        del self.response_callbacks[request_id]
                callback({'error': f"Помилка відправки запиту: {e}", 'request_id': request_id})

        return request_id

    def _delivery_callback(self, err, msg):
        '''
        Callback для підтвердження доставки повідомлення
        '''
        if err:
            logger.error(f"Помилка доставки повідомлення: {err}")
        else:
            logger.debug(f"Повідомлення доставлено в {msg.topic()} [{msg.partition()}] в оффсет {msg.offset()}")

    def _start_response_consumer(self):
        '''
        Запуск споживача для обробки відповідей
        '''
        response_consumer = Consumer({
            **self.consumer_config,
            'group.id': f"{self.consumer_group}-responses"
        })
        response_consumer.subscribe([self.response_topic])

        self.consumers.append(response_consumer)

        # Запуск обробника відповідей в окремому потоці
        response_thread = threading.Thread(
            target=self._response_handler,
            args=(response_consumer,),
            daemon=True
        )
        response_thread.start()
        self.worker_threads.append(response_thread)

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

            model_name = request.get('model_name')
            data = request.get('data')
            timestamp = request.get('timestamp', 0)

            # Перевірка часу очікування в черзі
            queue_time = time.time() - timestamp
            logger.debug(f"Запит {request_id} очікував у черзі {queue_time:.4f} секунд")

            # Пошук обробника для моделі
            handler = self.request_handlers.get(model_name)
            if not handler:
                raise ValueError(f"Обробник для моделі '{model_name}' не знайдено")

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

    def _response_handler(self, consumer):
        '''
        Обробник відповідей

        Параметри:
        -----------
        consumer: споживач Kafka
        '''
        try:
            while self.running:
                # Очищення прострочених колбеків
                self._clean_expired_callbacks()

                # Опитування наступного повідомлення
                msg = consumer.poll(1.0)

                if msg is None:
                    continue

                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        continue
                    else:
                        logger.error(f"Помилка споживача відповідей: {msg.error()}")
                        continue

                # Обробка відповіді
                try:
                    self._process_response(msg.key(), msg.value())
                except Exception as e:
                    logger.error(f"Помилка обробки відповіді: {e}")
        except Exception as e:
            logger.error(f"Критична помилка в обробнику відповідей: {e}")
        finally:
            logger.info("Обробник відповідей завершив роботу")

    def _process_response(self, key, value):
        '''
        Обробка відповіді інференсу

        Параметри:
        -----------
        key: ключ повідомлення (request_id)
        value: значення повідомлення (дані відповіді)
        '''
        try:
            # Декодування ключа та повідомлення
            request_id = key.decode('utf-8')
            response = json.loads(value.decode('utf-8'))

            # Пошук відповідного callback'у
            callback_info = None
            with self.response_lock:
                if request_id in self.response_callbacks:
                    callback_info = self.response_callbacks.pop(request_id)

            # Виклик callback'у, якщо він є
            if callback_info and 'callback' in callback_info:
                try:
                    callback_info['callback'](response)
                except Exception as e:
                    logger.error(f"Помилка виклику callback'у для {request_id}: {e}")

        except Exception as e:
            logger.error(f"Помилка обробки відповіді {key}: {e}")

    def _clean_expired_callbacks(self):
        '''
        Очищення прострочених callback'ів
        '''
        now = time.time()
        expired_ids = []

        with self.response_lock:
            for request_id, info in self.response_callbacks.items():
                if now > info.get('expires', 0):
                    expired_ids.append(request_id)

            for request_id in expired_ids:
                callback_info = self.response_callbacks.pop(request_id, None)
                if callback_info and 'callback' in callback_info:
                    try:
                        callback_info['callback']({
                            'request_id': request_id,
                            'success': False,
                            'error': 'Timeout waiting for response',
                            'timestamp': now
                        })
                    except Exception as e:
                        logger.error(f"Помилка виклику callback'у для простроченого запиту {request_id}: {e}")

# Приклад використання
if __name__ == "__main__":
    # Приклад обробника для моделі
    def example_handler(data):
        # Імітація інференсу
        time.sleep(0.5)  # Імітація затримки обробки
        return {'prediction': data * 2}

    # Колбек для обробки результату
    def result_callback(response):
        if response.get('success', False):
            print(f"Отримано результат: {response['result']}")
        else:
            print(f"Помилка: {response.get('error', 'Невідома помилка')}")

    # Створення та запуск сервісу
    service = KafkaQueueService(
        bootstrap_servers="localhost:9092",
        request_topic="model-inference-requests",
        response_topic="model-inference-responses",
        consumer_group="model-inference-group",
        num_workers=2
    )

    # Реєстрація обробника
    service.register_handler("example_model", example_handler)

    # Запуск сервісу
    service.start()

    try:
        # Відправка тестового запиту
        for i in range(5):
            request_id = service.submit_inference_request(
                model_name="example_model",
                data=i,
                callback=result_callback
            )
            print(f"Відправлено запит {request_id}")

        # Очікування завершення обробки
        time.sleep(10)
    finally:
        # Зупинка сервісу
        service.stop()
