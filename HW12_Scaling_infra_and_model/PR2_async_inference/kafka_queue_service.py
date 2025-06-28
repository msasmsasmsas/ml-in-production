# новлена версія для PR
#!/usr/bin/env python

'''
РЎРµСЂРІС–СЃ Р°СЃРёРЅС…СЂРѕРЅРЅРѕРіРѕ С–РЅС„РµСЂРµРЅСЃСѓ Р· РІРёРєРѕСЂРёСЃС‚Р°РЅРЅСЏРј Apache Kafka
'''

import os
import json
import time
import uuid
import threading
import logging
from typing import Dict, List, Any, Optional, Callable

# Kafka РєР»С–С”РЅС‚
from confluent_kafka import Producer, Consumer, KafkaError, KafkaException

# РќР°Р»Р°С€С‚СѓРІР°РЅРЅСЏ Р»РѕРіСѓРІР°РЅРЅСЏ
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('kafka_queue')

class KafkaQueueService:
    '''
    РЎРµСЂРІС–СЃ Р°СЃРёРЅС…СЂРѕРЅРЅРѕРіРѕ С–РЅС„РµСЂРµРЅСЃСѓ Р· РІРёРєРѕСЂРёСЃС‚Р°РЅРЅСЏРј Apache Kafka
    '''

    def __init__(self, 
                bootstrap_servers: str,
                request_topic: str,
                response_topic: str,
                consumer_group: str,
                num_workers: int = 2):
        '''
        Р†РЅС–С†С–Р°Р»С–Р·Р°С†С–СЏ СЃРµСЂРІС–СЃСѓ Kafka С‡РµСЂРіРё

        РџР°СЂР°РјРµС‚СЂРё:
        -----------
        bootstrap_servers: СЃРїРёСЃРѕРє Kafka СЃРµСЂРІРµСЂС–РІ
        request_topic: С‚РµРјР° РґР»СЏ Р·Р°РїРёС‚С–РІ С–РЅС„РµСЂРµРЅСЃСѓ
        response_topic: С‚РµРјР° РґР»СЏ РІС–РґРїРѕРІС–РґРµР№ С–РЅС„РµСЂРµРЅСЃСѓ
        consumer_group: РіСЂСѓРїР° СЃРїРѕР¶РёРІР°С‡С–РІ
        num_workers: РєС–Р»СЊРєС–СЃС‚СЊ РІРѕСЂРєРµСЂС–РІ РґР»СЏ РѕР±СЂРѕР±РєРё Р·Р°РїРёС‚С–РІ
        '''
        self.bootstrap_servers = bootstrap_servers
        self.request_topic = request_topic
        self.response_topic = response_topic
        self.consumer_group = consumer_group
        self.num_workers = num_workers

        # РќР°Р»Р°С€С‚СѓРІР°РЅРЅСЏ РїСЂРѕРґСЋСЃРµСЂР° Kafka
        self.producer_config = {
            'bootstrap.servers': bootstrap_servers,
            'client.id': f'model-inference-producer-{uuid.uuid4()}',
            'acks': 'all',  # РћС‡С–РєСѓРІР°С‚Рё РїС–РґС‚РІРµСЂРґР¶РµРЅРЅСЏ РІС–Рґ РІСЃС–С… СЂРµРїР»С–Рє
            'retries': 3,    # РљС–Р»СЊРєС–СЃС‚СЊ РїРѕРІС‚РѕСЂРЅРёС… СЃРїСЂРѕР±
            'retry.backoff.ms': 100  # Р—Р°С‚СЂРёРјРєР° РјС–Р¶ СЃРїСЂРѕР±Р°РјРё
        }

        # РќР°Р»Р°С€С‚СѓРІР°РЅРЅСЏ СЃРїРѕР¶РёРІР°С‡Р° Kafka
        self.consumer_config = {
            'bootstrap.servers': bootstrap_servers,
            'group.id': consumer_group,
            'auto.offset.reset': 'earliest',  # РџРѕС‡РёРЅР°С‚Рё Р· РЅР°Р№СЃС‚Р°СЂС–С€РёС… РїРѕРІС–РґРѕРјР»РµРЅСЊ
            'enable.auto.commit': True,       # РђРІС‚РѕРјР°С‚РёС‡РЅРёР№ РєРѕРјС–С‚ РѕС„СЃРµС‚С–РІ
            'auto.commit.interval.ms': 1000,  # Р†РЅС‚РµСЂРІР°Р» РєРѕРјС–С‚Сѓ РѕС„СЃРµС‚С–РІ
            'max.poll.interval.ms': 300000    # РњР°РєСЃРёРјР°Р»СЊРЅРёР№ С–РЅС‚РµСЂРІР°Р» РјС–Р¶ РѕРїРёС‚СѓРІР°РЅРЅСЏРјРё
        }

        self.producer = None
        self.consumers = []
        self.worker_threads = []
        self.running = False

        # РљРѕР»Р±РµРєРё РґР»СЏ РѕР±СЂРѕР±РєРё Р·Р°РїРёС‚С–РІ
        self.request_handlers = {}

        # РЎР»РѕРІРЅРёРє РґР»СЏ callback-С–РІ РѕС‡С–РєСѓРІР°РЅРЅСЏ РІС–РґРїРѕРІС–РґРµР№
        self.response_callbacks = {}
        self.response_lock = threading.Lock()

    def start(self):
        '''
        Р—Р°РїСѓСЃРє СЃРµСЂРІС–СЃСѓ Kafka С‡РµСЂРіРё
        '''
        if self.running:
            logger.warning("РЎРµСЂРІС–СЃ РІР¶Рµ Р·Р°РїСѓС‰РµРЅРѕ")
            return

        self.running = True

        # РЎС‚РІРѕСЂРµРЅРЅСЏ РїСЂРѕРґСЋСЃРµСЂР°
        self.producer = Producer(self.producer_config)

        # Р—Р°РїСѓСЃРє РѕР±СЂРѕР±РЅРёРєР° РІС–РґРїРѕРІС–РґРµР№
        self._start_response_consumer()

        # Р—Р°РїСѓСЃРє РІРѕСЂРєРµСЂС–РІ
        for i in range(self.num_workers):
            self._start_worker(i)

        logger.info(f"Р—Р°РїСѓС‰РµРЅРѕ СЃРµСЂРІС–СЃ Kafka С‡РµСЂРіРё Р· {self.num_workers} РІРѕСЂРєРµСЂР°РјРё")

    def stop(self):
        '''
        Р—СѓРїРёРЅРєР° СЃРµСЂРІС–СЃСѓ Kafka С‡РµСЂРіРё
        '''
        if not self.running:
            return

        self.running = False

        # Р—СѓРїРёРЅРєР° РІСЃС–С… РІРѕСЂРєРµСЂС–РІ
        for consumer in self.consumers:
            consumer.close()

        # РћС‡С–РєСѓРІР°РЅРЅСЏ Р·Р°РІРµСЂС€РµРЅРЅСЏ РІСЃС–С… РїРѕС‚РѕРєС–РІ
        for thread in self.worker_threads:
            thread.join(timeout=5.0)

        # Р—Р°РєСЂРёС‚С‚СЏ РїСЂРѕРґСЋСЃРµСЂР°
        if self.producer:
            self.producer.flush()

        self.consumers = []
        self.worker_threads = []
        self.response_callbacks = {}

        logger.info("РЎРµСЂРІС–СЃ Kafka С‡РµСЂРіРё Р·СѓРїРёРЅРµРЅРѕ")

    def register_handler(self, model_name: str, handler_func: Callable):
        '''
        Р РµС”СЃС‚СЂР°С†С–СЏ РѕР±СЂРѕР±РЅРёРєР° РґР»СЏ РјРѕРґРµР»С–

        РџР°СЂР°РјРµС‚СЂРё:
        -----------
        model_name: РЅР°Р·РІР° РјРѕРґРµР»С–
        handler_func: С„СѓРЅРєС†С–СЏ-РѕР±СЂРѕР±РЅРёРє Р·Р°РїРёС‚С–РІ С–РЅС„РµСЂРµРЅСЃСѓ
        '''
        self.request_handlers[model_name] = handler_func
        logger.info(f"Р—Р°СЂРµС”СЃС‚СЂРѕРІР°РЅРѕ РѕР±СЂРѕР±РЅРёРє РґР»СЏ РјРѕРґРµР»С– '{model_name}'")

    def submit_inference_request(self, model_name: str, data: Any, callback: Optional[Callable] = None, timeout: int = 30):
        '''
        Р’С–РґРїСЂР°РІРєР° Р·Р°РїРёС‚Сѓ РЅР° Р°СЃРёРЅС…СЂРѕРЅРЅРёР№ С–РЅС„РµСЂРµРЅСЃ

        РџР°СЂР°РјРµС‚СЂРё:
        -----------
        model_name: РЅР°Р·РІР° РјРѕРґРµР»С–
        data: РґР°РЅС– РґР»СЏ С–РЅС„РµСЂРµРЅСЃСѓ
        callback: С„СѓРЅРєС†С–СЏ, СЏРєР° Р±СѓРґРµ РІРёРєР»РёРєР°РЅР° Р· СЂРµР·СѓР»СЊС‚Р°С‚РѕРј
        timeout: С‚Р°Р№РјР°СѓС‚ РѕС‡С–РєСѓРІР°РЅРЅСЏ РІС–РґРїРѕРІС–РґС– РІ СЃРµРєСѓРЅРґР°С…

        РџРѕРІРµСЂС‚Р°С”:
        -----------
        request_id: С–РґРµРЅС‚РёС„С–РєР°С‚РѕСЂ Р·Р°РїРёС‚Сѓ
        '''
        request_id = str(uuid.uuid4())

        # РЎС‚РІРѕСЂРµРЅРЅСЏ РїРѕРІС–РґРѕРјР»РµРЅРЅСЏ
        message = {
            'request_id': request_id,
            'model_name': model_name,
            'timestamp': time.time(),
            'data': data
        }

        # РЎРµСЂС–Р°Р»С–Р·Р°С†С–СЏ РїРѕРІС–РґРѕРјР»РµРЅРЅСЏ
        try:
            payload = json.dumps(message).encode('utf-8')
        except (TypeError, ValueError) as e:
            logger.error(f"РџРѕРјРёР»РєР° СЃРµСЂС–Р°Р»С–Р·Р°С†С–С— Р·Р°РїРёС‚Сѓ: {e}")
            if callback:
                callback({'error': f"РџРѕРјРёР»РєР° СЃРµСЂС–Р°Р»С–Р·Р°С†С–С— Р·Р°РїРёС‚Сѓ: {e}", 'request_id': request_id})
            return request_id

        # Р РµС”СЃС‚СЂР°С†С–СЏ callback'Сѓ РґР»СЏ РѕС‡С–РєСѓРІР°РЅРЅСЏ РІС–РґРїРѕРІС–РґС–
        if callback:
            with self.response_lock:
                self.response_callbacks[request_id] = {
                    'callback': callback,
                    'expires': time.time() + timeout
                }

        # Р’С–РґРїСЂР°РІРєР° РїРѕРІС–РґРѕРјР»РµРЅРЅСЏ РІ Kafka
        try:
            self.producer.produce(
                self.request_topic,
                key=request_id.encode('utf-8'),
                value=payload,
                callback=self._delivery_callback
            )
            self.producer.poll(0)  # РўСЂРёРіРµСЂ РґРѕСЃС‚Р°РІРєРё РїРѕРІС–РґРѕРјР»РµРЅСЊ
        except Exception as e:
            logger.error(f"РџРѕРјРёР»РєР° РІС–РґРїСЂР°РІРєРё Р·Р°РїРёС‚Сѓ РІ Kafka: {e}")
            if callback:
                with self.response_lock:
                    if request_id in self.response_callbacks:
                        del self.response_callbacks[request_id]
                callback({'error': f"РџРѕРјРёР»РєР° РІС–РґРїСЂР°РІРєРё Р·Р°РїРёС‚Сѓ: {e}", 'request_id': request_id})

        return request_id

    def _delivery_callback(self, err, msg):
        '''
        Callback РґР»СЏ РїС–РґС‚РІРµСЂРґР¶РµРЅРЅСЏ РґРѕСЃС‚Р°РІРєРё РїРѕРІС–РґРѕРјР»РµРЅРЅСЏ
        '''
        if err:
            logger.error(f"РџРѕРјРёР»РєР° РґРѕСЃС‚Р°РІРєРё РїРѕРІС–РґРѕРјР»РµРЅРЅСЏ: {err}")
        else:
            logger.debug(f"РџРѕРІС–РґРѕРјР»РµРЅРЅСЏ РґРѕСЃС‚Р°РІР»РµРЅРѕ РІ {msg.topic()} [{msg.partition()}] РІ РѕС„С„СЃРµС‚ {msg.offset()}")

    def _start_response_consumer(self):
        '''
        Р—Р°РїСѓСЃРє СЃРїРѕР¶РёРІР°С‡Р° РґР»СЏ РѕР±СЂРѕР±РєРё РІС–РґРїРѕРІС–РґРµР№
        '''
        response_consumer = Consumer({
            **self.consumer_config,
            'group.id': f"{self.consumer_group}-responses"
        })
        response_consumer.subscribe([self.response_topic])

        self.consumers.append(response_consumer)

        # Р—Р°РїСѓСЃРє РѕР±СЂРѕР±РЅРёРєР° РІС–РґРїРѕРІС–РґРµР№ РІ РѕРєСЂРµРјРѕРјСѓ РїРѕС‚РѕС†С–
        response_thread = threading.Thread(
            target=self._response_handler,
            args=(response_consumer,),
            daemon=True
        )
        response_thread.start()
        self.worker_threads.append(response_thread)

    def _start_worker(self, worker_id):
        '''
        Р—Р°РїСѓСЃРє РІРѕСЂРєРµСЂР° РґР»СЏ РѕР±СЂРѕР±РєРё Р·Р°РїРёС‚С–РІ

        РџР°СЂР°РјРµС‚СЂРё:
        -----------
        worker_id: С–РґРµРЅС‚РёС„С–РєР°С‚РѕСЂ РІРѕСЂРєРµСЂР°
        '''
        # РЎС‚РІРѕСЂРµРЅРЅСЏ СЃРїРѕР¶РёРІР°С‡Р° РґР»СЏ РІРѕСЂРєРµСЂР°
        worker_consumer = Consumer({
            **self.consumer_config,
            'group.id': f"{self.consumer_group}-{worker_id}"
        })
        worker_consumer.subscribe([self.request_topic])

        self.consumers.append(worker_consumer)

        # Р—Р°РїСѓСЃРє РІРѕСЂРєРµСЂР° РІ РѕРєСЂРµРјРѕРјСѓ РїРѕС‚РѕС†С–
        worker_thread = threading.Thread(
            target=self._worker_loop,
            args=(worker_id, worker_consumer),
            daemon=True
        )
        worker_thread.start()
        self.worker_threads.append(worker_thread)

        logger.info(f"Р—Р°РїСѓС‰РµРЅРѕ РІРѕСЂРєРµСЂ {worker_id}")

    def _worker_loop(self, worker_id, consumer):
        '''
        РћСЃРЅРѕРІРЅРёР№ С†РёРєР» РІРѕСЂРєРµСЂР°

        РџР°СЂР°РјРµС‚СЂРё:
        -----------
        worker_id: С–РґРµРЅС‚РёС„С–РєР°С‚РѕСЂ РІРѕСЂРєРµСЂР°
        consumer: СЃРїРѕР¶РёРІР°С‡ Kafka
        '''
        try:
            while self.running:
                msg = consumer.poll(1.0)

                if msg is None:
                    continue

                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        # РљС–РЅРµС†СЊ РїР°СЂС‚РёС†С–С—
                        continue
                    else:
                        logger.error(f"РџРѕРјРёР»РєР° СЃРїРѕР¶РёРІР°С‡Р°: {msg.error()}")
                        continue

                # РћР±СЂРѕР±РєР° РїРѕРІС–РґРѕРјР»РµРЅРЅСЏ
                try:
                    self._process_request(msg.key(), msg.value(), worker_id)
                except Exception as e:
                    logger.error(f"РџРѕРјРёР»РєР° РѕР±СЂРѕР±РєРё Р·Р°РїРёС‚Сѓ РІРѕСЂРєРµСЂРѕРј {worker_id}: {e}")
        except Exception as e:
            logger.error(f"РљСЂРёС‚РёС‡РЅР° РїРѕРјРёР»РєР° РІ РІРѕСЂРєРµСЂС– {worker_id}: {e}")
        finally:
            logger.info(f"Р’РѕСЂРєРµСЂ {worker_id} Р·Р°РІРµСЂС€РёРІ СЂРѕР±РѕС‚Сѓ")

    def _process_request(self, key, value, worker_id):
        '''
        РћР±СЂРѕР±РєР° Р·Р°РїРёС‚Сѓ С–РЅС„РµСЂРµРЅСЃСѓ

        РџР°СЂР°РјРµС‚СЂРё:
        -----------
        key: РєР»СЋС‡ РїРѕРІС–РґРѕРјР»РµРЅРЅСЏ (request_id)
        value: Р·РЅР°С‡РµРЅРЅСЏ РїРѕРІС–РґРѕРјР»РµРЅРЅСЏ (РґР°РЅС– Р·Р°РїРёС‚Сѓ)
        worker_id: С–РґРµРЅС‚РёС„С–РєР°С‚РѕСЂ РІРѕСЂРєРµСЂР°
        '''
        start_time = time.time()

        try:
            # Р”РµРєРѕРґСѓРІР°РЅРЅСЏ РєР»СЋС‡Р° С‚Р° РїРѕРІС–РґРѕРјР»РµРЅРЅСЏ
            request_id = key.decode('utf-8')
            request = json.loads(value.decode('utf-8'))

            model_name = request.get('model_name')
            data = request.get('data')
            timestamp = request.get('timestamp', 0)

            # РџРµСЂРµРІС–СЂРєР° С‡Р°СЃСѓ РѕС‡С–РєСѓРІР°РЅРЅСЏ РІ С‡РµСЂР·С–
            queue_time = time.time() - timestamp
            logger.debug(f"Р—Р°РїРёС‚ {request_id} РѕС‡С–РєСѓРІР°РІ Сѓ С‡РµСЂР·С– {queue_time:.4f} СЃРµРєСѓРЅРґ")

            # РџРѕС€СѓРє РѕР±СЂРѕР±РЅРёРєР° РґР»СЏ РјРѕРґРµР»С–
            handler = self.request_handlers.get(model_name)
            if not handler:
                raise ValueError(f"РћР±СЂРѕР±РЅРёРє РґР»СЏ РјРѕРґРµР»С– '{model_name}' РЅРµ Р·РЅР°Р№РґРµРЅРѕ")

            # Р’РёРєР»РёРє РѕР±СЂРѕР±РЅРёРєР°
            result = handler(data)

            # РџС–РґРіРѕС‚РѕРІРєР° РІС–РґРїРѕРІС–РґС–
            response = {
                'request_id': request_id,
                'success': True,
                'timestamp': time.time(),
                'queue_time': queue_time,
                'processing_time': time.time() - start_time,
                'result': result
            }

        except Exception as e:
            logger.error(f"РџРѕРјРёР»РєР° РѕР±СЂРѕР±РєРё Р·Р°РїРёС‚Сѓ {key}: {e}")
            response = {
                'request_id': key.decode('utf-8') if isinstance(key, bytes) else str(key),
                'success': False,
                'timestamp': time.time(),
                'error': str(e)
            }

        # Р’С–РґРїСЂР°РІРєР° РІС–РґРїРѕРІС–РґС–
        try:
            self.producer.produce(
                self.response_topic,
                key=key,
                value=json.dumps(response).encode('utf-8'),
                callback=self._delivery_callback
            )
            self.producer.poll(0)  # РўСЂРёРіРµСЂ РґРѕСЃС‚Р°РІРєРё РїРѕРІС–РґРѕРјР»РµРЅСЊ
        except Exception as e:
            logger.error(f"РџРѕРјРёР»РєР° РІС–РґРїСЂР°РІРєРё РІС–РґРїРѕРІС–РґС–: {e}")

    def _response_handler(self, consumer):
        '''
        РћР±СЂРѕР±РЅРёРє РІС–РґРїРѕРІС–РґРµР№

        РџР°СЂР°РјРµС‚СЂРё:
        -----------
        consumer: СЃРїРѕР¶РёРІР°С‡ Kafka
        '''
        try:
            while self.running:
                # РћС‡РёС‰РµРЅРЅСЏ РїСЂРѕСЃС‚СЂРѕС‡РµРЅРёС… РєРѕР»Р±РµРєС–РІ
                self._clean_expired_callbacks()

                # РћРїРёС‚СѓРІР°РЅРЅСЏ РЅР°СЃС‚СѓРїРЅРѕРіРѕ РїРѕРІС–РґРѕРјР»РµРЅРЅСЏ
                msg = consumer.poll(1.0)

                if msg is None:
                    continue

                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        continue
                    else:
                        logger.error(f"РџРѕРјРёР»РєР° СЃРїРѕР¶РёРІР°С‡Р° РІС–РґРїРѕРІС–РґРµР№: {msg.error()}")
                        continue

                # РћР±СЂРѕР±РєР° РІС–РґРїРѕРІС–РґС–
                try:
                    self._process_response(msg.key(), msg.value())
                except Exception as e:
                    logger.error(f"РџРѕРјРёР»РєР° РѕР±СЂРѕР±РєРё РІС–РґРїРѕРІС–РґС–: {e}")
        except Exception as e:
            logger.error(f"РљСЂРёС‚РёС‡РЅР° РїРѕРјРёР»РєР° РІ РѕР±СЂРѕР±РЅРёРєСѓ РІС–РґРїРѕРІС–РґРµР№: {e}")
        finally:
            logger.info("РћР±СЂРѕР±РЅРёРє РІС–РґРїРѕРІС–РґРµР№ Р·Р°РІРµСЂС€РёРІ СЂРѕР±РѕС‚Сѓ")

    def _process_response(self, key, value):
        '''
        РћР±СЂРѕР±РєР° РІС–РґРїРѕРІС–РґС– С–РЅС„РµСЂРµРЅСЃСѓ

        РџР°СЂР°РјРµС‚СЂРё:
        -----------
        key: РєР»СЋС‡ РїРѕРІС–РґРѕРјР»РµРЅРЅСЏ (request_id)
        value: Р·РЅР°С‡РµРЅРЅСЏ РїРѕРІС–РґРѕРјР»РµРЅРЅСЏ (РґР°РЅС– РІС–РґРїРѕРІС–РґС–)
        '''
        try:
            # Р”РµРєРѕРґСѓРІР°РЅРЅСЏ РєР»СЋС‡Р° С‚Р° РїРѕРІС–РґРѕРјР»РµРЅРЅСЏ
            request_id = key.decode('utf-8')
            response = json.loads(value.decode('utf-8'))

            # РџРѕС€СѓРє РІС–РґРїРѕРІС–РґРЅРѕРіРѕ callback'Сѓ
            callback_info = None
            with self.response_lock:
                if request_id in self.response_callbacks:
                    callback_info = self.response_callbacks.pop(request_id)

            # Р’РёРєР»РёРє callback'Сѓ, СЏРєС‰Рѕ РІС–РЅ С”
            if callback_info and 'callback' in callback_info:
                try:
                    callback_info['callback'](response)
                except Exception as e:
                    logger.error(f"РџРѕРјРёР»РєР° РІРёРєР»РёРєСѓ callback'Сѓ РґР»СЏ {request_id}: {e}")

        except Exception as e:
            logger.error(f"РџРѕРјРёР»РєР° РѕР±СЂРѕР±РєРё РІС–РґРїРѕРІС–РґС– {key}: {e}")

    def _clean_expired_callbacks(self):
        '''
        РћС‡РёС‰РµРЅРЅСЏ РїСЂРѕСЃС‚СЂРѕС‡РµРЅРёС… callback'С–РІ
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
                        logger.error(f"РџРѕРјРёР»РєР° РІРёРєР»РёРєСѓ callback'Сѓ РґР»СЏ РїСЂРѕСЃС‚СЂРѕС‡РµРЅРѕРіРѕ Р·Р°РїРёС‚Сѓ {request_id}: {e}")

# РџСЂРёРєР»Р°Рґ РІРёРєРѕСЂРёСЃС‚Р°РЅРЅСЏ
if __name__ == "__main__":
    # РџСЂРёРєР»Р°Рґ РѕР±СЂРѕР±РЅРёРєР° РґР»СЏ РјРѕРґРµР»С–
    def example_handler(data):
        # Р†РјС–С‚Р°С†С–СЏ С–РЅС„РµСЂРµРЅСЃСѓ
        time.sleep(0.5)  # Р†РјС–С‚Р°С†С–СЏ Р·Р°С‚СЂРёРјРєРё РѕР±СЂРѕР±РєРё
        return {'prediction': data * 2}

    # РљРѕР»Р±РµРє РґР»СЏ РѕР±СЂРѕР±РєРё СЂРµР·СѓР»СЊС‚Р°С‚Сѓ
    def result_callback(response):
        if response.get('success', False):
            print(f"РћС‚СЂРёРјР°РЅРѕ СЂРµР·СѓР»СЊС‚Р°С‚: {response['result']}")
        else:
            print(f"РџРѕРјРёР»РєР°: {response.get('error', 'РќРµРІС–РґРѕРјР° РїРѕРјРёР»РєР°')}")

    # РЎС‚РІРѕСЂРµРЅРЅСЏ С‚Р° Р·Р°РїСѓСЃРє СЃРµСЂРІС–СЃСѓ
    service = KafkaQueueService(
        bootstrap_servers="localhost:9092",
        request_topic="model-inference-requests",
        response_topic="model-inference-responses",
        consumer_group="model-inference-group",
        num_workers=2
    )

    # Р РµС”СЃС‚СЂР°С†С–СЏ РѕР±СЂРѕР±РЅРёРєР°
    service.register_handler("example_model", example_handler)

    # Р—Р°РїСѓСЃРє СЃРµСЂРІС–СЃСѓ
    service.start()

    try:
        # Р’С–РґРїСЂР°РІРєР° С‚РµСЃС‚РѕРІРѕРіРѕ Р·Р°РїРёС‚Сѓ
        for i in range(5):
            request_id = service.submit_inference_request(
                model_name="example_model",
                data=i,
                callback=result_callback
            )
            print(f"Р’С–РґРїСЂР°РІР»РµРЅРѕ Р·Р°РїРёС‚ {request_id}")

        # РћС‡С–РєСѓРІР°РЅРЅСЏ Р·Р°РІРµСЂС€РµРЅРЅСЏ РѕР±СЂРѕР±РєРё
        time.sleep(10)
    finally:
        # Р—СѓРїРёРЅРєР° СЃРµСЂРІС–СЃСѓ
        service.stop()

