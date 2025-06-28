# новлена версія для PR
#!/usr/bin/env python

'''
Р’РѕСЂРєРµСЂ РґР»СЏ Р°СЃРёРЅС…СЂРѕРЅРЅРѕРіРѕ С–РЅС„РµСЂРµРЅСЃСѓ РјРѕРґРµР»РµР№
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

# Kafka С–РЅС‚РµРіСЂР°С†С–СЏ
from confluent_kafka import Producer, Consumer, KafkaError, KafkaException

# РќР°Р»Р°С€С‚СѓРІР°РЅРЅСЏ Р»РѕРіСѓРІР°РЅРЅСЏ
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('model_worker')

class ModelInferenceWorker:
    '''
    Р’РѕСЂРєРµСЂ РґР»СЏ Р°СЃРёРЅС…СЂРѕРЅРЅРѕРіРѕ С–РЅС„РµСЂРµРЅСЃСѓ РјРѕРґРµР»РµР№
    '''

    def __init__(self, 
                bootstrap_servers: str,
                request_topic: str,
                response_topic: str,
                consumer_group: str,
                model_path: str,
                num_workers: int = 2):
        '''
        Р†РЅС–С†С–Р°Р»С–Р·Р°С†С–СЏ РІРѕСЂРєРµСЂР°

        РџР°СЂР°РјРµС‚СЂРё:
        -----------
        bootstrap_servers: СЃРїРёСЃРѕРє Kafka СЃРµСЂРІРµСЂС–РІ
        request_topic: С‚РµРјР° РґР»СЏ Р·Р°РїРёС‚С–РІ С–РЅС„РµСЂРµРЅСЃСѓ
        response_topic: С‚РµРјР° РґР»СЏ РІС–РґРїРѕРІС–РґРµР№ С–РЅС„РµСЂРµРЅСЃСѓ
        consumer_group: РіСЂСѓРїР° СЃРїРѕР¶РёРІР°С‡С–РІ
        model_path: С€Р»СЏС… РґРѕ С„Р°Р№Р»Сѓ РјРѕРґРµР»С–
        num_workers: РєС–Р»СЊРєС–СЃС‚СЊ РІРѕСЂРєРµСЂС–РІ РґР»СЏ РѕР±СЂРѕР±РєРё Р·Р°РїРёС‚С–РІ
        '''
        self.bootstrap_servers = bootstrap_servers
        self.request_topic = request_topic
        self.response_topic = response_topic
        self.consumer_group = consumer_group
        self.model_path = model_path
        self.num_workers = num_workers

        # РќР°Р»Р°С€С‚СѓРІР°РЅРЅСЏ РїСЂРѕРґСЋСЃРµСЂР° Kafka
        self.producer_config = {
            'bootstrap.servers': bootstrap_servers,
            'client.id': f'model-worker-producer',
            'acks': 'all',
            'retries': 3,
            'retry.backoff.ms': 100
        }

        # РќР°Р»Р°С€С‚СѓРІР°РЅРЅСЏ СЃРїРѕР¶РёРІР°С‡Р° Kafka
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

        # Р—Р°РІР°РЅС‚Р°Р¶РµРЅРЅСЏ РјРѕРґРµР»С–
        self.model = self._load_model(model_path)

        # РћР±СЂРѕР±РЅРёРєРё РјРѕРґРµР»РµР№ Р·Р° РЅР°Р·РІР°РјРё
        self.model_handlers = {
            'default': self._default_inference,
            'image_classification': self._image_classification_inference,
            'text_classification': self._text_classification_inference
        }

    def _load_model(self, model_path):
        '''
        Р—Р°РІР°РЅС‚Р°Р¶РµРЅРЅСЏ РјРѕРґРµР»С–

        РџР°СЂР°РјРµС‚СЂРё:
        -----------
        model_path: С€Р»СЏС… РґРѕ С„Р°Р№Р»Сѓ РјРѕРґРµР»С–

        РџРѕРІРµСЂС‚Р°С”:
        -----------
        Р·Р°РІР°РЅС‚Р°Р¶РµРЅР° РјРѕРґРµР»СЊ
        '''
        try:
            logger.info(f"Р—Р°РІР°РЅС‚Р°Р¶РµРЅРЅСЏ РјРѕРґРµР»С– Р· {model_path}")

            # Р’РёР·РЅР°С‡РµРЅРЅСЏ С‚РёРїСѓ РјРѕРґРµР»С– Р·Р° СЂРѕР·С€РёСЂРµРЅРЅСЏРј
            if model_path.endswith('.pt') or model_path.endswith('.pth'):
                # PyTorch РјРѕРґРµР»СЊ
                if torch.cuda.is_available():
                    logger.info("Р’РёРєРѕСЂРёСЃС‚Р°РЅРЅСЏ GPU РґР»СЏ С–РЅС„РµСЂРµРЅСЃСѓ")
                    device = torch.device("cuda")
                else:
                    logger.info("Р’РёРєРѕСЂРёСЃС‚Р°РЅРЅСЏ CPU РґР»СЏ С–РЅС„РµСЂРµРЅСЃСѓ")
                    device = torch.device("cpu")

                # Р—Р°РІР°РЅС‚Р°Р¶РµРЅРЅСЏ РјРѕРґРµР»С– PyTorch
                model = torch.load(model_path, map_location=device)
                model.eval()  # РџРµСЂРµРјРёРєР°РЅРЅСЏ РІ СЂРµР¶РёРј С–РЅС„РµСЂРµРЅСЃСѓ

                return model

            elif model_path.endswith('.onnx'):
                # ONNX РјРѕРґРµР»СЊ
                import onnxruntime as ort

                # РќР°Р»Р°С€С‚СѓРІР°РЅРЅСЏ СЃРµСЃС–С— ONNX Runtime
                sess_options = ort.SessionOptions()
                sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

                # Р’РёР±С–СЂ РїСЂРѕРІР°Р№РґРµСЂР°
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider']

                # РЎС‚РІРѕСЂРµРЅРЅСЏ СЃРµСЃС–С—
                session = ort.InferenceSession(model_path, sess_options, providers=providers)

                return session

            else:
                # Р—Р°РіР»СѓС€РєР° РґР»СЏ С‚РµСЃС‚СѓРІР°РЅРЅСЏ
                logger.warning(f"Р¤РѕСЂРјР°С‚ РјРѕРґРµР»С– РЅРµ СЂРѕР·РїС–Р·РЅР°РЅРѕ, РІРёРєРѕСЂРёСЃС‚Р°РЅРЅСЏ Р·Р°РіР»СѓС€РєРё РґР»СЏ С‚РµСЃС‚СѓРІР°РЅРЅСЏ")
                return DummyModel()

        except Exception as e:
            logger.error(f"РџРѕРјРёР»РєР° Р·Р°РІР°РЅС‚Р°Р¶РµРЅРЅСЏ РјРѕРґРµР»С–: {e}")
            # Р—Р°РіР»СѓС€РєР° РґР»СЏ С‚РµСЃС‚СѓРІР°РЅРЅСЏ
            return DummyModel()

    def start(self):
        '''
        Р—Р°РїСѓСЃРє РІРѕСЂРєРµСЂР°
        '''
        if self.running:
            logger.warning("Р’РѕСЂРєРµСЂ РІР¶Рµ Р·Р°РїСѓС‰РµРЅРѕ")
            return

        self.running = True

        # РЎС‚РІРѕСЂРµРЅРЅСЏ РїСЂРѕРґСЋСЃРµСЂР°
        self.producer = Producer(self.producer_config)

        # Р—Р°РїСѓСЃРє РІРѕСЂРєРµСЂС–РІ
        for i in range(self.num_workers):
            self._start_worker(i)

        logger.info(f"Р—Р°РїСѓС‰РµРЅРѕ {self.num_workers} РІРѕСЂРєРµСЂС–РІ РґР»СЏ РѕР±СЂРѕР±РєРё Р·Р°РїРёС‚С–РІ")

    def stop(self):
        '''
        Р—СѓРїРёРЅРєР° РІРѕСЂРєРµСЂР°
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

        logger.info("Р’РѕСЂРєРµСЂ Р·СѓРїРёРЅРµРЅРѕ")

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

            model_name = request.get('model_name', 'default')
            data = request.get('data')
            timestamp = request.get('timestamp', 0)

            # РџРµСЂРµРІС–СЂРєР° С‡Р°СЃСѓ РѕС‡С–РєСѓРІР°РЅРЅСЏ РІ С‡РµСЂР·С–
            queue_time = time.time() - timestamp
            logger.debug(f"Р—Р°РїРёС‚ {request_id} РѕС‡С–РєСѓРІР°РІ Сѓ С‡РµСЂР·С– {queue_time:.4f} СЃРµРєСѓРЅРґ")

            # Р’РёР±С–СЂ РѕР±СЂРѕР±РЅРёРєР° РЅР° РѕСЃРЅРѕРІС– РЅР°Р·РІРё РјРѕРґРµР»С–
            handler = self.model_handlers.get(model_name, self.model_handlers['default'])

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

    def _delivery_callback(self, err, msg):
        '''
        Callback РґР»СЏ РїС–РґС‚РІРµСЂРґР¶РµРЅРЅСЏ РґРѕСЃС‚Р°РІРєРё РїРѕРІС–РґРѕРјР»РµРЅРЅСЏ
        '''
        if err:
            logger.error(f"РџРѕРјРёР»РєР° РґРѕСЃС‚Р°РІРєРё РїРѕРІС–РґРѕРјР»РµРЅРЅСЏ: {err}")
        else:
            logger.debug(f"РџРѕРІС–РґРѕРјР»РµРЅРЅСЏ РґРѕСЃС‚Р°РІР»РµРЅРѕ РІ {msg.topic()} [{msg.partition()}] РІ РѕС„С„СЃРµС‚ {msg.offset()}")

    def _default_inference(self, data):
        '''
        РЎС‚Р°РЅРґР°СЂС‚РЅРёР№ РѕР±СЂРѕР±РЅРёРє С–РЅС„РµСЂРµРЅСЃСѓ

        РџР°СЂР°РјРµС‚СЂРё:
        -----------
        data: РґР°РЅС– РґР»СЏ С–РЅС„РµСЂРµРЅСЃСѓ

        РџРѕРІРµСЂС‚Р°С”:
        -----------
        СЂРµР·СѓР»СЊС‚Р°С‚ С–РЅС„РµСЂРµРЅСЃСѓ
        '''
        logger.info(f"Р’РёРєРѕРЅР°РЅРЅСЏ СЃС‚Р°РЅРґР°СЂС‚РЅРѕРіРѕ С–РЅС„РµСЂРµРЅСЃСѓ")

        # РџРµСЂРµРІС–СЂРєР° С‚РёРїСѓ РґР°РЅРёС…
        if isinstance(data, dict) and 'content_type' in data and 'data' in data:
            # РЎРїСЂРѕР±Р° РѕР±СЂРѕР±РёС‚Рё РЅР° РѕСЃРЅРѕРІС– content_type
            content_type = data['content_type']

            if 'image' in content_type:
                return self._image_classification_inference(data)
            elif 'text' in content_type:
                return self._text_classification_inference(data)

        # Р’РёРєРѕСЂРёСЃС‚Р°РЅРЅСЏ РјРѕРґРµР»С– РЅР°РїСЂСЏРјСѓ
        try:
            if hasattr(self.model, 'predict'):
                return self.model.predict(data).tolist()
            elif hasattr(self.model, 'forward'):
                with torch.no_grad():
                    input_tensor = torch.tensor(data).float()
                    output = self.model(input_tensor)
                    return output.cpu().numpy().tolist()
            else:
                # РџРѕРІРµСЂС‚Р°С”РјРѕ РґР°РЅС– РґР»СЏ Р·Р°РіР»СѓС€РєРё
                return {"prediction": "default", "score": 0.9, "processing_time": time.time() - start_time}
        except Exception as e:
            logger.error(f"РџРѕРјРёР»РєР° СЃС‚Р°РЅРґР°СЂС‚РЅРѕРіРѕ С–РЅС„РµСЂРµРЅСЃСѓ: {e}")
            raise

    def _image_classification_inference(self, data):
        '''
        РћР±СЂРѕР±РЅРёРє С–РЅС„РµСЂРµРЅСЃСѓ РґР»СЏ РєР»Р°СЃРёС„С–РєР°С†С–С— Р·РѕР±СЂР°Р¶РµРЅСЊ

        РџР°СЂР°РјРµС‚СЂРё:
        -----------
        data: РґР°РЅС– РґР»СЏ С–РЅС„РµСЂРµРЅСЃСѓ

        РџРѕРІРµСЂС‚Р°С”:
        -----------
        СЂРµР·СѓР»СЊС‚Р°С‚ С–РЅС„РµСЂРµРЅСЃСѓ
        '''
        logger.info(f"Р’РёРєРѕРЅР°РЅРЅСЏ С–РЅС„РµСЂРµРЅСЃСѓ РєР»Р°СЃРёС„С–РєР°С†С–С— Р·РѕР±СЂР°Р¶РµРЅСЊ")
        start_time = time.time()

        try:
            # РћР±СЂРѕР±РєР° РІС…С–РґРЅРёС… РґР°РЅРёС…
            if isinstance(data, dict) and 'data' in data:
                # РљРѕРЅРІРµСЂС‚Р°С†С–СЏ Р· hex Сѓ Р±Р°Р№С‚Рё
                image_bytes = binascii.unhexlify(data['data'])
                image = Image.open(io.BytesIO(image_bytes))
            elif isinstance(data, str):
                # РЎРїСЂРѕР±Р° Р·Р°РІР°РЅС‚Р°Р¶РёС‚Рё Р· base64
                import base64
                image_bytes = base64.b64decode(data)
                image = Image.open(io.BytesIO(image_bytes))
            else:
                raise ValueError("РќРµРїС–РґС‚СЂРёРјСѓРІР°РЅРёР№ С„РѕСЂРјР°С‚ РґР°РЅРёС… РґР»СЏ РєР»Р°СЃРёС„С–РєР°С†С–С— Р·РѕР±СЂР°Р¶РµРЅСЊ")

            # РџРѕРїРµСЂРµРґРЅСЏ РѕР±СЂРѕР±РєР° Р·РѕР±СЂР°Р¶РµРЅРЅСЏ
            from torchvision import transforms

            preprocess = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

            image_tensor = preprocess(image).unsqueeze(0)

            # РџРµСЂРµРЅРµСЃРµРЅРЅСЏ РЅР° РїРѕС‚СЂС–Р±РЅРёР№ РїСЂРёСЃС‚СЂС–Р№
            device = next(self.model.parameters()).device
            image_tensor = image_tensor.to(device)

            # Р’РёРєРѕРЅР°РЅРЅСЏ С–РЅС„РµСЂРµРЅСЃСѓ
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]

                # РћС‚СЂРёРјР°РЅРЅСЏ С‚РѕРї-5 РєР»Р°СЃС–РІ
                top5_prob, top5_catid = torch.topk(probabilities, 5)

                # РљРѕРЅРІРµСЂС‚Р°С†С–СЏ СЂРµР·СѓР»СЊС‚Р°С‚С–РІ
                result = {
                    "predictions": [
                        {"class_id": int(idx), "score": float(prob)} 
                        for idx, prob in zip(top5_catid.cpu().numpy(), top5_prob.cpu().numpy())
                    ],
                    "processing_time": time.time() - start_time
                }

                return result

        except Exception as e:
            logger.error(f"РџРѕРјРёР»РєР° С–РЅС„РµСЂРµРЅСЃСѓ РєР»Р°СЃРёС„С–РєР°С†С–С— Р·РѕР±СЂР°Р¶РµРЅСЊ: {e}")
            raise

    def _text_classification_inference(self, data):
        '''
        РћР±СЂРѕР±РЅРёРє С–РЅС„РµСЂРµРЅСЃСѓ РґР»СЏ РєР»Р°СЃРёС„С–РєР°С†С–С— С‚РµРєСЃС‚Сѓ

        РџР°СЂР°РјРµС‚СЂРё:
        -----------
        data: РґР°РЅС– РґР»СЏ С–РЅС„РµСЂРµРЅСЃСѓ

        РџРѕРІРµСЂС‚Р°С”:
        -----------
        СЂРµР·СѓР»СЊС‚Р°С‚ С–РЅС„РµСЂРµРЅСЃСѓ
        '''
        logger.info(f"Р’РёРєРѕРЅР°РЅРЅСЏ С–РЅС„РµСЂРµРЅСЃСѓ РєР»Р°СЃРёС„С–РєР°С†С–С— С‚РµРєСЃС‚Сѓ")
        start_time = time.time()

        try:
            # РћР±СЂРѕР±РєР° РІС…С–РґРЅРёС… РґР°РЅРёС…
            if isinstance(data, dict) and 'data' in data:
                # РўРµРєСЃС‚ РјРѕР¶Рµ Р±СѓС‚Рё Сѓ С„РѕСЂРјР°С‚С– UTF-8 hex
                if isinstance(data['data'], str):
                    try:
                        text = binascii.unhexlify(data['data']).decode('utf-8')
                    except:
                        text = data['data']  # Р’Р¶Рµ С‚РµРєСЃС‚
                else:
                    text = str(data['data'])
            elif isinstance(data, str):
                text = data
            else:
                raise ValueError("РќРµРїС–РґС‚СЂРёРјСѓРІР°РЅРёР№ С„РѕСЂРјР°С‚ РґР°РЅРёС… РґР»СЏ РєР»Р°СЃРёС„С–РєР°С†С–С— С‚РµРєСЃС‚Сѓ")

            # Р—Р°РіР»СѓС€РєР° РґР»СЏ С‚РµСЃС‚СѓРІР°РЅРЅСЏ
            result = {
                "predictions": [
                    {"label": "positive", "score": 0.8},
                    {"label": "negative", "score": 0.2}
                ],
                "processing_time": time.time() - start_time
            }

            return result

        except Exception as e:
            logger.error(f"РџРѕРјРёР»РєР° С–РЅС„РµСЂРµРЅСЃСѓ РєР»Р°СЃРёС„С–РєР°С†С–С— С‚РµРєСЃС‚Сѓ: {e}")
            raise

class DummyModel:
    '''
    Р—Р°РіР»СѓС€РєР° РјРѕРґРµР»С– РґР»СЏ С‚РµСЃС‚СѓРІР°РЅРЅСЏ
    '''
    def __init__(self):
        self.device = "cpu"

    def __call__(self, input_data):
        # Р†РјС–С‚Р°С†С–СЏ РѕР±СЂРѕР±РєРё РґР»СЏ С‚РµСЃС‚СѓРІР°РЅРЅСЏ
        time.sleep(0.1)  # Р†РјС–С‚Р°С†С–СЏ Р·Р°С‚СЂРёРјРєРё С–РЅС„РµСЂРµРЅСЃСѓ

        if isinstance(input_data, torch.Tensor):
            # Р”Р»СЏ РєР»Р°СЃРёС„С–РєР°С†С–С— Р·РѕР±СЂР°Р¶РµРЅСЊ
            batch_size = input_data.shape[0]
            return torch.rand(batch_size, 1000)  # Р†РјС–С‚Р°С†С–СЏ РІРёС…РѕРґСѓ РєР»Р°СЃРёС„С–РєР°С‚РѕСЂР° Р· 1000 РєР»Р°СЃР°РјРё
        else:
            # Р”Р»СЏ С‚РµРєСЃС‚РѕРІРѕС— РєР»Р°СЃРёС„С–РєР°С†С–С— Р°Р±Рѕ С–РЅС€РѕРіРѕ С‚РёРїСѓ РґР°РЅРёС…
            return {"prediction": "dummy", "score": 0.9}

    def predict(self, data):
        # Р”Р»СЏ scikit-learn С–РЅС‚РµСЂС„РµР№СЃСѓ
        return np.random.rand(len(data) if hasattr(data, '__len__') else 1)

    def parameters(self):
        # Р†РјС–С‚Р°С†С–СЏ РїР°СЂР°РјРµС‚СЂС–РІ РјРѕРґРµР»С–
        yield torch.nn.Parameter(torch.randn(1))

def main():
    parser = argparse.ArgumentParser(description="Р’РѕСЂРєРµСЂ РґР»СЏ Р°СЃРёРЅС…СЂРѕРЅРЅРѕРіРѕ С–РЅС„РµСЂРµРЅСЃСѓ РјРѕРґРµР»РµР№")
    parser.add_argument("--bootstrap-servers", type=str, default=os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"),
                        help="РЎРїРёСЃРѕРє Kafka СЃРµСЂРІРµСЂС–РІ")
    parser.add_argument("--request-topic", type=str, default=os.environ.get("KAFKA_REQUEST_TOPIC", "model-inference-requests"),
                        help="РўРµРјР° РґР»СЏ Р·Р°РїРёС‚С–РІ С–РЅС„РµСЂРµРЅСЃСѓ")
    parser.add_argument("--response-topic", type=str, default=os.environ.get("KAFKA_RESPONSE_TOPIC", "model-inference-responses"),
                        help="РўРµРјР° РґР»СЏ РІС–РґРїРѕРІС–РґРµР№ С–РЅС„РµСЂРµРЅСЃСѓ")
    parser.add_argument("--consumer-group", type=str, default=os.environ.get("KAFKA_CONSUMER_GROUP", "model-inference-worker"),
                        help="Р“СЂСѓРїР° СЃРїРѕР¶РёРІР°С‡С–РІ")
    parser.add_argument("--model-path", type=str, default=os.environ.get("MODEL_PATH", "models/model.pt"),
                        help="РЁР»СЏС… РґРѕ С„Р°Р№Р»Сѓ РјРѕРґРµР»С–")
    parser.add_argument("--num-workers", type=int, default=int(os.environ.get("NUM_WORKERS", "2")),
                        help="РљС–Р»СЊРєС–СЃС‚СЊ РІРѕСЂРєРµСЂС–РІ РґР»СЏ РѕР±СЂРѕР±РєРё Р·Р°РїРёС‚С–РІ")

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
        # РћС‡С–РєСѓРІР°РЅРЅСЏ РЅР° СЃРёРіРЅР°Р» Р·Р°РІРµСЂС€РµРЅРЅСЏ
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("РћС‚СЂРёРјР°РЅРѕ СЃРёРіРЅР°Р» Р·Р°РІРµСЂС€РµРЅРЅСЏ")
    finally:
        worker.stop()

if __name__ == "__main__":
    main()

