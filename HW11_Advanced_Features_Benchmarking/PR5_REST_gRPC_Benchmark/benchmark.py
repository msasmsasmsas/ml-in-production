# Updated version for PR
#!/usr/bin/env python
"""
РЎРєСЂРёРїС‚ РґР»СЏ РїРѕСЂС–РІРЅСЏР»СЊРЅРѕРіРѕ Р±РµРЅС‡РјР°СЂРєС–РЅРіСѓ REST С‚Р° gRPC С–РЅС‚РµСЂС„РµР№СЃС–РІ
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

# РќР°Р»Р°С€С‚СѓРІР°РЅРЅСЏ Р»РѕРіСѓРІР°РЅРЅСЏ
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('benchmark')

# Р”Р»СЏ REST Р·Р°РїРёС‚С–РІ
import requests

# Р”Р»СЏ gRPC Р·Р°РїРёС‚С–РІ
try:
    import grpc
    import inference_pb2
    import inference_pb2_grpc
    grpc_available = True
except ImportError:
    grpc_available = False
    logger.warning("gRPC РјРѕРґСѓР»С– РЅРµ Р·РЅР°Р№РґРµРЅРѕ. gRPC Р±РµРЅС‡РјР°СЂРєС–РЅРі Р±СѓРґРµ РЅРµРґРѕСЃС‚СѓРїРЅРёР№.")

class RestClient:
    """
    РљР»С–С”РЅС‚ РґР»СЏ REST API
    """
    def __init__(self, base_url, timeout=30):
        """
        Р†РЅС–С†С–Р°Р»С–Р·Р°С†С–СЏ РєР»С–С”РЅС‚Р°

        РџР°СЂР°РјРµС‚СЂРё:
        -----------
        base_url: Р±Р°Р·РѕРІР° URL Р°РґСЂРµСЃР° СЃРµСЂРІРµСЂР°
        timeout: С‚Р°Р№РјР°СѓС‚ РґР»СЏ Р·Р°РїРёС‚С–РІ
        """
        self.base_url = base_url
        self.predict_url = f"{base_url}/predict"
        self.health_url = f"{base_url}/health"
        self.timeout = timeout

    def check_health(self):
        """
        РџРµСЂРµРІС–СЂРєР° СЃС‚Р°РЅСѓ СЃРµСЂРІРµСЂР°

        РџРѕРІРµСЂС‚Р°С”:
        -----------
        (СЃС‚Р°С‚СѓСЃ, РїРѕРІС–РґРѕРјР»РµРЅРЅСЏ)
        """
        try:
            response = requests.get(self.health_url, timeout=self.timeout)
            if response.status_code == 200:
                return True, response.json()
            else:
                return False, f"HTTP РїРѕРјРёР»РєР°: {response.status_code}"
        except Exception as e:
            return False, str(e)

    def predict(self, image_path):
        """
        Р’С–РґРїСЂР°РІР»СЏС” Р·Р°РїРёС‚ РЅР° РїСЂРѕРіРЅРѕР·СѓРІР°РЅРЅСЏ

        РџР°СЂР°РјРµС‚СЂРё:
        -----------
        image_path: С€Р»СЏС… РґРѕ С„Р°Р№Р»Сѓ Р·РѕР±СЂР°Р¶РµРЅРЅСЏ

        РџРѕРІРµСЂС‚Р°С”:
        -----------
        РєРѕСЂС‚РµР¶ (РІС–РґРїРѕРІС–РґСЊ, С‡Р°СЃ РІРёРєРѕРЅР°РЅРЅСЏ, СѓСЃРїС–С…)
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
                    return {'error': f'HTTP РїРѕРјРёР»РєР°: {response.status_code}'}, elapsed, False
        except Exception as e:
            return {'error': str(e)}, 0, False

class GrpcClient:
    """
    РљР»С–С”РЅС‚ РґР»СЏ gRPC API
    """
    def __init__(self, server_address, timeout=30):
        """
        Р†РЅС–С†С–Р°Р»С–Р·Р°С†С–СЏ РєР»С–С”РЅС‚Р°

        РџР°СЂР°РјРµС‚СЂРё:
        -----------
        server_address: Р°РґСЂРµСЃР° gRPC СЃРµСЂРІРµСЂР°
        timeout: С‚Р°Р№РјР°СѓС‚ РґР»СЏ Р·Р°РїРёС‚С–РІ
        """
        if not grpc_available:
            raise ImportError("gRPC РјРѕРґСѓР»С– РЅРµРґРѕСЃС‚СѓРїРЅС–")

        # РЎС‚РІРѕСЂРµРЅРЅСЏ РєР°РЅР°Р»Сѓ Р· РѕРїС†С–СЏРјРё РґР»СЏ РІРµР»РёРєРёС… РїРѕРІС–РґРѕРјР»РµРЅСЊ
        channel_options = [
            ('grpc.max_send_message_length', 50 * 1024 * 1024),  # 50 MB
            ('grpc.max_receive_message_length', 50 * 1024 * 1024),  # 50 MB
        ]
        self.channel = grpc.insecure_channel(server_address, options=channel_options)
        self.stub = inference_pb2_grpc.InferenceServiceStub(self.channel)
        self.timeout = timeout

    def check_health(self):
        """
        РџРµСЂРµРІС–СЂРєР° СЃС‚Р°РЅСѓ СЃРµСЂРІРµСЂР°

        РџРѕРІРµСЂС‚Р°С”:
        -----------
        (СЃС‚Р°С‚СѓСЃ, РїРѕРІС–РґРѕРјР»РµРЅРЅСЏ)
        """
        try:
            request = inference_pb2.HealthCheckRequest()
            response = self.stub.HealthCheck(request, timeout=self.timeout)

            if response.status == inference_pb2.ServingStatus.SERVING:
                return True, {'status': 'ok', 'metadata': dict(response.metadata)}
            else:
                return False, f"РЎРµСЂРІРµСЂ РЅРµ РіРѕС‚РѕРІРёР№: {response.status}"
        except Exception as e:
            return False, str(e)

    def predict(self, image_path):
        """
        Р’С–РґРїСЂР°РІР»СЏС” Р·Р°РїРёС‚ РЅР° РїСЂРѕРіРЅРѕР·СѓРІР°РЅРЅСЏ

        РџР°СЂР°РјРµС‚СЂРё:
        -----------
        image_path: С€Р»СЏС… РґРѕ С„Р°Р№Р»Сѓ Р·РѕР±СЂР°Р¶РµРЅРЅСЏ

        РџРѕРІРµСЂС‚Р°С”:
        -----------
        РєРѕСЂС‚РµР¶ (РІС–РґРїРѕРІС–РґСЊ, С‡Р°СЃ РІРёРєРѕРЅР°РЅРЅСЏ, СѓСЃРїС–С…)
        """
        try:
            # Р—С‡РёС‚СѓРІР°РЅРЅСЏ С„Р°Р№Р»Сѓ Р·РѕР±СЂР°Р¶РµРЅРЅСЏ
            with open(image_path, 'rb') as f:
                image_data = f.read()

            # РЎС‚РІРѕСЂРµРЅРЅСЏ Р·Р°РїРёС‚Сѓ
            request = inference_pb2.PredictRequest(
                data=image_data,
                content_type='image/jpeg'
            )

            # Р’РёРјС–СЂСЋРІР°РЅРЅСЏ С‡Р°СЃСѓ РІРёРєРѕРЅР°РЅРЅСЏ Р·Р°РїРёС‚Сѓ
            start_time = time.time()
            response = self.stub.Predict(request, timeout=self.timeout)
            elapsed = time.time() - start_time

            # РљРѕРЅРІРµСЂС‚Р°С†С–СЏ response Сѓ СЃР»РѕРІРЅРёРє РґР»СЏ СѓРЅС–С„С–РєР°С†С–С— Р· REST
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
        Р—Р°РєСЂРёС‚С‚СЏ Р·'С”РґРЅР°РЅРЅСЏ
        """
        self.channel.close()

class BenchmarkSuite:
    """
    РќР°Р±С–СЂ С–РЅСЃС‚СЂСѓРјРµРЅС‚С–РІ РґР»СЏ Р±РµРЅС‡РјР°СЂРєС–РЅРіСѓ REST С‚Р° gRPC
    """
    def __init__(self, rest_url=None, grpc_server=None, timeout=30):
        """
        Р†РЅС–С†С–Р°Р»С–Р·Р°С†С–СЏ РЅР°Р±РѕСЂСѓ Р±РµРЅС‡РјР°СЂРєС–РЅРіСѓ

        РџР°СЂР°РјРµС‚СЂРё:
        -----------
        rest_url: URL REST API
        grpc_server: Р°РґСЂРµСЃР° gRPC СЃРµСЂРІРµСЂР°
        timeout: С‚Р°Р№РјР°СѓС‚ РґР»СЏ Р·Р°РїРёС‚С–РІ
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
                logger.error(f"РџРѕРјРёР»РєР° С–РЅС–С†С–Р°Р»С–Р·Р°С†С–С— gRPC РєР»С–С”РЅС‚Р°: {e}")

    def check_server_health(self):
        """
        РџРµСЂРµРІС–СЂРєР° СЃС‚Р°РЅСѓ СЃРµСЂРІРµСЂС–РІ

        РџРѕРІРµСЂС‚Р°С”:
        -----------
        (rest_status, grpc_status)
        """
        rest_status = (False, "REST РєР»С–С”РЅС‚ РЅРµ С–РЅС–С†С–Р°Р»С–Р·РѕРІР°РЅРѕ")
        grpc_status = (False, "gRPC РєР»С–С”РЅС‚ РЅРµ С–РЅС–С†С–Р°Р»С–Р·РѕРІР°РЅРѕ")

        if self.rest_client:
            rest_status = self.rest_client.check_health()
            logger.info(f"REST СЃРµСЂРІРµСЂ: {'РіРѕС‚РѕРІРёР№' if rest_status[0] else 'РЅРµ РіРѕС‚РѕРІРёР№'} - {rest_status[1]}")

        if self.grpc_client:
            grpc_status = self.grpc_client.check_health()
            logger.info(f"gRPC СЃРµСЂРІРµСЂ: {'РіРѕС‚РѕРІРёР№' if grpc_status[0] else 'РЅРµ РіРѕС‚РѕРІРёР№'} - {grpc_status[1]}")

        return rest_status, grpc_status

    def run_benchmark(self, protocol, image_path, num_requests, concurrency):
        """
        Р—Р°РїСѓСЃРєР°С” Р±РµРЅС‡РјР°СЂРєС–РЅРі РґР»СЏ РІРєР°Р·Р°РЅРѕРіРѕ РїСЂРѕС‚РѕРєРѕР»Сѓ

        РџР°СЂР°РјРµС‚СЂРё:
        -----------
        protocol: РїСЂРѕС‚РѕРєРѕР» ('rest' Р°Р±Рѕ 'grpc')
        image_path: С€Р»СЏС… РґРѕ С„Р°Р№Р»Сѓ Р·РѕР±СЂР°Р¶РµРЅРЅСЏ
        num_requests: РєС–Р»СЊРєС–СЃС‚СЊ Р·Р°РїРёС‚С–РІ
        concurrency: СЂС–РІРµРЅСЊ РїР°СЂР°Р»РµР»С–Р·РјСѓ

        РџРѕРІРµСЂС‚Р°С”:
        -----------
        СЃР»РѕРІРЅРёРє Р· СЂРµР·СѓР»СЊС‚Р°С‚Р°РјРё
        """
        if protocol == 'rest' and self.rest_client:
            return self._run_protocol_benchmark(self.rest_client, image_path, num_requests, concurrency)
        elif protocol == 'grpc' and self.grpc_client:
            return self._run_protocol_benchmark(self.grpc_client, image_path, num_requests, concurrency)
        else:
            logger.error(f"РќРµРјРѕР¶Р»РёРІРѕ Р·Р°РїСѓСЃС‚РёС‚Рё Р±РµРЅС‡РјР°СЂРєС–РЅРі РґР»СЏ РїСЂРѕС‚РѕРєРѕР»Сѓ {protocol}")
            return None

    def _run_protocol_benchmark(self, client, image_path, num_requests, concurrency):
        """
        Р—Р°РїСѓСЃРєР°С” Р±РµРЅС‡РјР°СЂРєС–РЅРі РґР»СЏ РєРѕРЅРєСЂРµС‚РЅРѕРіРѕ РєР»С–С”РЅС‚Р°

        РџР°СЂР°РјРµС‚СЂРё:
        -----------
        client: РєР»С–С”РЅС‚ (RestClient Р°Р±Рѕ GrpcClient)
        image_path: С€Р»СЏС… РґРѕ С„Р°Р№Р»Сѓ Р·РѕР±СЂР°Р¶РµРЅРЅСЏ
        num_requests: РєС–Р»СЊРєС–СЃС‚СЊ Р·Р°РїРёС‚С–РІ
        concurrency: СЂС–РІРµРЅСЊ РїР°СЂР°Р»РµР»С–Р·РјСѓ

        РџРѕРІРµСЂС‚Р°С”:
        -----------
        СЃР»РѕРІРЅРёРє Р· СЂРµР·СѓР»СЊС‚Р°С‚Р°РјРё
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
                # Р—Р±РµСЂРµР¶РµРЅРЅСЏ С‡Р°СЃСѓ РѕР±СЂРѕР±РєРё РЅР° СЃРµСЂРІРµСЂС–, СЏРєС‰Рѕ РґРѕСЃС‚СѓРїРЅРѕ
                if 'processing_time' in response:
                    server_times.append(response['processing_time'] / 1000)  # РєРѕРЅРІРµСЂС‚Р°С†С–СЏ Р· РјСЃ Сѓ СЃРµРєСѓРЅРґРё

            return {'response': response, 'elapsed': elapsed, 'success': success}

        logger.info(f"Р—Р°РїСѓСЃРє {num_requests} Р·Р°РїРёС‚С–РІ Р· РїР°СЂР°Р»РµР»С–Р·РјРѕРј {concurrency}...")
        start_time = time.time()

        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [executor.submit(send_request) for _ in range(num_requests)]

            for i, future in enumerate(concurrent.futures.as_completed(futures)):
                result = future.result()
                results.append(result)

                # Р›РѕРіСѓРІР°РЅРЅСЏ РїСЂРѕРіСЂРµСЃСѓ
                if (i+1) % max(1, num_requests // 10) == 0 or i+1 == num_requests:
                    logger.info(f"Р—Р°РІРµСЂС€РµРЅРѕ {i+1}/{num_requests} Р·Р°РїРёС‚С–РІ")

        total_time = time.time() - start_time

        # РћР±С‡РёСЃР»РµРЅРЅСЏ СЃС‚Р°С‚РёСЃС‚РёРєРё
        latencies = [r['elapsed'] for r in results if r['success']]

        if not latencies:
            logger.error("Р’СЃС– Р·Р°РїРёС‚Рё Р·Р°РІРµСЂС€РёР»РёСЃСЏ Р· РїРѕРјРёР»РєР°РјРё")
            return {
                'total_requests': num_requests,
                'successful_requests': 0,
                'failed_requests': num_requests,
                'total_time': total_time,
                'concurrency': concurrency,
                'stats': None
            }

        # Р‘Р°Р·РѕРІР° СЃС‚Р°С‚РёСЃС‚РёРєР°
        stats = {
            'min': min(latencies) * 1000,  # РјСЃ
            'max': max(latencies) * 1000,  # РјСЃ
            'mean': statistics.mean(latencies) * 1000,  # РјСЃ
            'median': statistics.median(latencies) * 1000,  # РјСЃ
            'p90': np.percentile(latencies, 90) * 1000,  # РјСЃ
            'p95': np.percentile(latencies, 95) * 1000,  # РјСЃ
            'p99': np.percentile(latencies, 99) * 1000,  # РјСЃ
            'std': statistics.stdev(latencies) * 1000 if len(latencies) > 1 else 0,  # РјСЃ
            'rps': len(latencies) / total_time  # Р·Р°РїРёС‚С–РІ РЅР° СЃРµРєСѓРЅРґСѓ
        }

        # РЎС‚Р°С‚РёСЃС‚РёРєР° С‡Р°СЃСѓ СЃРµСЂРІРµСЂР°, СЏРєС‰Рѕ РґРѕСЃС‚СѓРїРЅР°
        if server_times:
            stats['server_time'] = {
                'min': min(server_times) * 1000,  # РјСЃ
                'max': max(server_times) * 1000,  # РјСЃ
                'mean': statistics.mean(server_times) * 1000,  # РјСЃ
                'median': statistics.median(server_times) * 1000,  # РјСЃ
                'p90': np.percentile(server_times, 90) * 1000,  # РјСЃ
                'p95': np.percentile(server_times, 95) * 1000,  # РјСЃ
                'p99': np.percentile(server_times, 99) * 1000  # РјСЃ
            }

            # РћР±С‡РёСЃР»РµРЅРЅСЏ РјРµСЂРµР¶РµРІРѕС— Р·Р°С‚СЂРёРјРєРё
            network_times = [l - s for l, s in zip(latencies, server_times)]
            stats['network_time'] = {
                'min': min(network_times) * 1000,  # РјСЃ
                'max': max(network_times) * 1000,  # РјСЃ
                'mean': statistics.mean(network_times) * 1000,  # РјСЃ
                'median': statistics.median(network_times) * 1000,  # РјСЃ
                'p90': np.percentile(network_times, 90) * 1000,  # РјСЃ
                'p95': np.percentile(network_times, 95) * 1000,  # РјСЃ
                'p99': np.percentile(network_times, 99) * 1000  # РјСЃ
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
        Р—Р°РїСѓСЃРєР°С” РїРѕСЂС–РІРЅСЏР»СЊРЅРёР№ Р±РµРЅС‡РјР°СЂРєС–РЅРі REST С‚Р° gRPC

        РџР°СЂР°РјРµС‚СЂРё:
        -----------
        image_path: С€Р»СЏС… РґРѕ С„Р°Р№Р»Сѓ Р·РѕР±СЂР°Р¶РµРЅРЅСЏ
        num_requests: РєС–Р»СЊРєС–СЃС‚СЊ Р·Р°РїРёС‚С–РІ
        concurrency: СЂС–РІРµРЅСЊ РїР°СЂР°Р»РµР»С–Р·РјСѓ

        РџРѕРІРµСЂС‚Р°С”:
        -----------
        (rest_results, grpc_results)
        """
        rest_results = None
        grpc_results = None

        # РџРµСЂРµРІС–СЂРєР° СЃС‚Р°РЅСѓ СЃРµСЂРІРµСЂС–РІ
        rest_status, grpc_status = self.check_server_health()

        # REST Р±РµРЅС‡РјР°СЂРєС–РЅРі
        if self.rest_client and rest_status[0]:
            logger.info("Р—Р°РїСѓСЃРє REST Р±РµРЅС‡РјР°СЂРєС–РЅРіСѓ...")
            rest_results = self.run_benchmark('rest', image_path, num_requests, concurrency)
        else:
            logger.warning("REST Р±РµРЅС‡РјР°СЂРєС–РЅРі РїСЂРѕРїСѓС‰РµРЅРѕ (СЃРµСЂРІРµСЂ РЅРµ РіРѕС‚РѕРІРёР№)")

        # gRPC Р±РµРЅС‡РјР°СЂРєС–РЅРі
        if self.grpc_client and grpc_status[0]:
            logger.info("Р—Р°РїСѓСЃРє gRPC Р±РµРЅС‡РјР°СЂРєС–РЅРіСѓ...")
            grpc_results = self.run_benchmark('grpc', image_path, num_requests, concurrency)
        else:
            logger.warning("gRPC Р±РµРЅС‡РјР°СЂРєС–РЅРі РїСЂРѕРїСѓС‰РµРЅРѕ (СЃРµСЂРІРµСЂ РЅРµ РіРѕС‚РѕРІРёР№)")

        return rest_results, grpc_results

    def close(self):
        """
        Р—Р°РєСЂРёС‚С‚СЏ РєР»С–С”РЅС‚С–РІ
        """
        if self.grpc_client:
            self.grpc_client.close()

def print_benchmark_results(protocol, results):
    """
    Р’РёРІРѕРґРёС‚СЊ СЂРµР·СѓР»СЊС‚Р°С‚Рё Р±РµРЅС‡РјР°СЂРєС–РЅРіСѓ

    РџР°СЂР°РјРµС‚СЂРё:
    -----------
    protocol: РЅР°Р·РІР° РїСЂРѕС‚РѕРєРѕР»Сѓ (REST Р°Р±Рѕ gRPC)
    results: СЃР»РѕРІРЅРёРє Р· СЂРµР·СѓР»СЊС‚Р°С‚Р°РјРё
    """
    if not results:
        logger.warning(f"Р РµР·СѓР»СЊС‚Р°С‚Рё {protocol} РІС–РґСЃСѓС‚РЅС–")
        return

    logger.info(f"\nР РµР·СѓР»СЊС‚Р°С‚Рё Р±РµРЅС‡РјР°СЂРєС–РЅРіСѓ {protocol}:")
    logger.info(f"Р—Р°РіР°Р»СЊРЅРёР№ С‡Р°СЃ: {results['total_time']:.2f} СЃ")
    logger.info(f"Р—Р°РїРёС‚С–РІ: {results['total_requests']}")
    logger.info(f"РЈСЃРїС–С€РЅРёС…: {results['successful_requests']} ({100 * results['successful_requests'] / results['total_requests']:.2f}%)")
    logger.info(f"РќРµРІРґР°Р»РёС…: {results['failed_requests']}")
    logger.info(f"Р С–РІРµРЅСЊ РїР°СЂР°Р»РµР»С–Р·РјСѓ: {results['concurrency']}")

    if results['stats']:
        stats = results['stats']
        logger.info("\nРЎС‚Р°С‚РёСЃС‚РёРєР° С‡Р°СЃСѓ РІРёРєРѕРЅР°РЅРЅСЏ (РјСЃ):")
        logger.info(f"  РњС–РЅ: {stats['min']:.2f}")
        logger.info(f"  РњР°РєСЃ: {stats['max']:.2f}")
        logger.info(f"  РЎРµСЂРµРґРЅС”: {stats['mean']:.2f}")
        logger.info(f"  РњРµРґС–Р°РЅР°: {stats['median']:.2f}")
        logger.info(f"  P90: {stats['p90']:.2f}")
        logger.info(f"  P95: {stats['p95']:.2f}")
        logger.info(f"  P99: {stats['p99']:.2f}")
        logger.info(f"  РЎС‚Р°РЅРґР°СЂС‚РЅРµ РІС–РґС…РёР»РµРЅРЅСЏ: {stats['std']:.2f}")
        logger.info(f"  Р—Р°РїРёС‚С–РІ РЅР° СЃРµРєСѓРЅРґСѓ (RPS): {stats['rps']:.2f}")

        if 'server_time' in stats:
            logger.info("\nР§Р°СЃ РѕР±СЂРѕР±РєРё РЅР° СЃРµСЂРІРµСЂС– (РјСЃ):")
            logger.info(f"  РњС–РЅ: {stats['server_time']['min']:.2f}")
            logger.info(f"  РњР°РєСЃ: {stats['server_time']['max']:.2f}")
            logger.info(f"  РЎРµСЂРµРґРЅС”: {stats['server_time']['mean']:.2f}")
            logger.info(f"  РњРµРґС–Р°РЅР°: {stats['server_time']['median']:.2f}")
            logger.info(f"  P95: {stats['server_time']['p95']:.2f}")

        if 'network_time' in stats:
            logger.info("\nРњРµСЂРµР¶РµРІР° Р·Р°С‚СЂРёРјРєР° (РјСЃ):")
            logger.info(f"  РњС–РЅ: {stats['network_time']['min']:.2f}")
            logger.info(f"  РњР°РєСЃ: {stats['network_time']['max']:.2f}")
            logger.info(f"  РЎРµСЂРµРґРЅС”: {stats['network_time']['mean']:.2f}")
            logger.info(f"  РњРµРґС–Р°РЅР°: {stats['network_time']['median']:.2f}")
            logger.info(f"  P95: {stats['network_time']['p95']:.2f}")

def compare_and_plot(rest_results, grpc_results, output_file=None):
    """
    РџРѕСЂС–РІРЅСЋС” С‚Р° РІС–Р·СѓР°Р»С–Р·СѓС” СЂРµР·СѓР»СЊС‚Р°С‚Рё Р±РµРЅС‡РјР°СЂРєС–РЅРіСѓ REST С‚Р° gRPC

    РџР°СЂР°РјРµС‚СЂРё:
    -----------
    rest_results: СЃР»РѕРІРЅРёРє Р· СЂРµР·СѓР»СЊС‚Р°С‚Р°РјРё REST
    grpc_results: СЃР»РѕРІРЅРёРє Р· СЂРµР·СѓР»СЊС‚Р°С‚Р°РјРё gRPC
    output_file: С€Р»СЏС… РґРѕ РІРёС…С–РґРЅРѕРіРѕ С„Р°Р№Р»Сѓ (СЏРєС‰Рѕ None, РіСЂР°С„С–РєРё РІС–РґРѕР±СЂР°Р¶Р°СЋС‚СЊСЃСЏ)

    РџРѕРІРµСЂС‚Р°С”:
    -----------
    True, СЏРєС‰Рѕ РІС–Р·СѓР°Р»С–Р·Р°С†С–СЏ СѓСЃРїС–С€РЅР°
    """
    if not rest_results or not rest_results['stats'] or \
       not grpc_results or not grpc_results['stats']:
        logger.error("РќРµРґРѕСЃС‚Р°С‚РЅСЊРѕ РґР°РЅРёС… РґР»СЏ РїРѕСЂС–РІРЅСЏРЅРЅСЏ")
        return False

    # РЎС‚РІРѕСЂРµРЅРЅСЏ РґР°С‚Р°С„СЂРµР№РјСѓ РґР»СЏ РІС–Р·СѓР°Р»С–Р·Р°С†С–С—
    rest_latencies = [l * 1000 for l in rest_results['raw_latencies']]  # РјСЃ
    grpc_latencies = [l * 1000 for l in grpc_results['raw_latencies']]  # РјСЃ

    rest_df = pd.DataFrame({'latency': rest_latencies, 'protocol': 'REST'})
    grpc_df = pd.DataFrame({'latency': grpc_latencies, 'protocol': 'gRPC'})
    df = pd.concat([rest_df, grpc_df])

    # РќР°Р»Р°С€С‚СѓРІР°РЅРЅСЏ СЃС‚РёР»СЋ РіСЂР°С„С–РєС–РІ
    sns.set(style="whitegrid")

    # РЎС‚РІРѕСЂРµРЅРЅСЏ РіСЂР°С„С–РєС–РІ
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('РџРѕСЂС–РІРЅСЏРЅРЅСЏ REST С‚Р° gRPC', fontsize=16)

    # Р“СЂР°С„С–Рє 1: RPS
    protocols = ['REST', 'gRPC']
    rps = [rest_results['stats']['rps'], grpc_results['stats']['rps']]

    axes[0, 0].bar(protocols, rps, color=['#3498db', '#2ecc71'])
    axes[0, 0].set_title('Р—Р°РїРёС‚Рё РЅР° СЃРµРєСѓРЅРґСѓ (RPS)')
    axes[0, 0].set_ylabel('RPS')
    for i, v in enumerate(rps):
        axes[0, 0].text(i, v, f"{v:.2f}", ha='center', va='bottom')

    # Р“СЂР°С„С–Рє 2: Р РѕР·РїРѕРґС–Р» Р·Р°С‚СЂРёРјРѕРє
    sns.boxplot(x='protocol', y='latency', data=df, ax=axes[0, 1], 
                palette={'REST': '#3498db', 'gRPC': '#2ecc71'})
    axes[0, 1].set_title('Р РѕР·РїРѕРґС–Р» Р·Р°С‚СЂРёРјРѕРє')
    axes[0, 1].set_ylabel('Р—Р°С‚СЂРёРјРєР° (РјСЃ)')

    # Р“СЂР°С„С–Рє 3: Р“С–СЃС‚РѕРіСЂР°РјР° Р·Р°С‚СЂРёРјРѕРє
    sns.histplot(data=df, x='latency', hue='protocol', kde=True, ax=axes[1, 0],
                 palette={'REST': '#3498db', 'gRPC': '#2ecc71'})
    axes[1, 0].set_title('Р“С–СЃС‚РѕРіСЂР°РјР° Р·Р°С‚СЂРёРјРѕРє')
    axes[1, 0].set_xlabel('Р—Р°С‚СЂРёРјРєР° (РјСЃ)')
    axes[1, 0].set_ylabel('РљС–Р»СЊРєС–СЃС‚СЊ Р·Р°РїРёС‚С–РІ')

    # Р“СЂР°С„С–Рє 4: РџРѕСЂС–РІРЅСЏРЅРЅСЏ РјРµС‚СЂРёРє
    metrics = ['mean', 'median', 'p90', 'p95', 'p99']
    rest_metrics = [rest_results['stats'][m] for m in metrics]
    grpc_metrics = [grpc_results['stats'][m] for m in metrics]

    x = np.arange(len(metrics))
    width = 0.35

    axes[1, 1].bar(x - width/2, rest_metrics, width, label='REST', color='#3498db')
    axes[1, 1].bar(x + width/2, grpc_metrics, width, label='gRPC', color='#2ecc71')

    axes[1, 1].set_title('РџРѕСЂС–РІРЅСЏРЅРЅСЏ РјРµС‚СЂРёРє')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(metrics)
    axes[1, 1].set_ylabel('Р—Р°С‚СЂРёРјРєР° (РјСЃ)')
    axes[1, 1].legend()

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if output_file:
        plt.savefig(output_file)
        logger.info(f"Р“СЂР°С„С–РєРё Р·Р±РµСЂРµР¶РµРЅРѕ Сѓ {output_file}")
    else:
        plt.show()

    return True

def save_results_json(rest_results, grpc_results, output_file):
    """
    Р—Р±РµСЂС–РіР°С” СЂРµР·СѓР»СЊС‚Р°С‚Рё Сѓ JSON С„Р°Р№Р»

    РџР°СЂР°РјРµС‚СЂРё:
    -----------
    rest_results: СЃР»РѕРІРЅРёРє Р· СЂРµР·СѓР»СЊС‚Р°С‚Р°РјРё REST
    grpc_results: СЃР»РѕРІРЅРёРє Р· СЂРµР·СѓР»СЊС‚Р°С‚Р°РјРё gRPC
    output_file: С€Р»СЏС… РґРѕ РІРёС…С–РґРЅРѕРіРѕ С„Р°Р№Р»Сѓ
    """
    # РЎС‚РІРѕСЂРµРЅРЅСЏ РєРѕРїС–Р№ Р±РµР· raw_latencies РґР»СЏ РєРѕРјРїР°РєС‚РЅРѕСЃС‚С–
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

    logger.info(f"Р РµР·СѓР»СЊС‚Р°С‚Рё Р·Р±РµСЂРµР¶РµРЅРѕ Сѓ {output_file}")

def main():
    parser = argparse.ArgumentParser(description='РџРѕСЂС–РІРЅСЏР»СЊРЅРёР№ Р±РµРЅС‡РјР°СЂРєС–РЅРі REST С‚Р° gRPC')

    # РћСЃРЅРѕРІРЅС– РїР°СЂР°РјРµС‚СЂРё
    parser.add_argument('--image', type=str, required=True,
                        help='РЁР»СЏС… РґРѕ С‚РµСЃС‚РѕРІРѕРіРѕ Р·РѕР±СЂР°Р¶РµРЅРЅСЏ')
    parser.add_argument('--rest-url', type=str, default='http://localhost:5000',
                        help='URL РґР»СЏ REST API')
    parser.add_argument('--grpc-server', type=str, default='localhost:50051',
                        help='РђРґСЂРµСЃР° gRPC СЃРµСЂРІРµСЂР°')

    # РџР°СЂР°РјРµС‚СЂРё Р±РµРЅС‡РјР°СЂРєС–РЅРіСѓ
    parser.add_argument('--requests', type=int, default=100,
                        help='РљС–Р»СЊРєС–СЃС‚СЊ Р·Р°РїРёС‚С–РІ')
    parser.add_argument('--concurrency', type=int, default=10,
                        help='Р С–РІРµРЅСЊ РїР°СЂР°Р»РµР»С–Р·РјСѓ')
    parser.add_argument('--timeout', type=int, default=30,
                        help='РўР°Р№РјР°СѓС‚ РґР»СЏ Р·Р°РїРёС‚С–РІ (СЃРµРєСѓРЅРґРё)')

    # РџР°СЂР°РјРµС‚СЂРё РІРёС…С–РґРЅРёС… РґР°РЅРёС…
    parser.add_argument('--output-json', type=str, default=None,
                        help='РЁР»СЏС… РґР»СЏ Р·Р±РµСЂРµР¶РµРЅРЅСЏ СЂРµР·СѓР»СЊС‚Р°С‚С–РІ Сѓ JSON')
    parser.add_argument('--output-plot', type=str, default=None,
                        help='РЁР»СЏС… РґР»СЏ Р·Р±РµСЂРµР¶РµРЅРЅСЏ РіСЂР°С„С–РєС–РІ')

    args = parser.parse_args()

    # РџРµСЂРµРІС–СЂРєР° РЅР°СЏРІРЅРѕСЃС‚С– С„Р°Р№Р»Сѓ Р·РѕР±СЂР°Р¶РµРЅРЅСЏ
    if not os.path.isfile(args.image):
        logger.error(f"РџРѕРјРёР»РєР°: С„Р°Р№Р» {args.image} РЅРµ С–СЃРЅСѓС”")
        return 1

    # Р†РЅС–С†С–Р°Р»С–Р·Р°С†С–СЏ РЅР°Р±РѕСЂСѓ Р±РµРЅС‡РјР°СЂРєС–РЅРіСѓ
    suite = BenchmarkSuite(args.rest_url, args.grpc_server, args.timeout)

    try:
        # Р—Р°РїСѓСЃРє РїРѕСЂС–РІРЅСЏР»СЊРЅРѕРіРѕ Р±РµРЅС‡РјР°СЂРєС–РЅРіСѓ
        rest_results, grpc_results = suite.run_comparison(
            args.image, args.requests, args.concurrency
        )

        # Р’РёРІРµРґРµРЅРЅСЏ СЂРµР·СѓР»СЊС‚Р°С‚С–РІ
        print_benchmark_results("REST", rest_results)
        print_benchmark_results("gRPC", grpc_results)

        # РџРѕСЂС–РІРЅСЏРЅРЅСЏ С‚Р° РІС–Р·СѓР°Р»С–Р·Р°С†С–СЏ
        if rest_results and grpc_results:
            compare_and_plot(rest_results, grpc_results, args.output_plot)

        # Р—Р±РµСЂРµР¶РµРЅРЅСЏ СЂРµР·СѓР»СЊС‚Р°С‚С–РІ
        if args.output_json:
            save_results_json(rest_results, grpc_results, args.output_json)

    finally:
        # Р—Р°РєСЂРёС‚С‚СЏ РєР»С–С”РЅС‚С–РІ
        suite.close()

    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())

