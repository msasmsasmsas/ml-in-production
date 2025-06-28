# Updated version for PR
#!/usr/bin/env python
"""
Р†РЅСЃС‚СЂСѓРјРµРЅС‚ РґР»СЏ Р±РµРЅС‡РјР°СЂРєС–РЅРіСѓ СЃРµСЂРІРµСЂС–РІ РјРѕРґРµР»РµР№ РјР°С€РёРЅРЅРѕРіРѕ РЅР°РІС‡Р°РЅРЅСЏ
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

# РЎРїСЂРѕР±СѓС”РјРѕ С–РјРїРѕСЂС‚СѓРІР°С‚Рё gRPC РјРѕРґСѓР»С–, СЏРєС‰Рѕ РІРѕРЅРё С”
try:
    import inference_pb2
    import inference_pb2_grpc
    grpc_available = True
except ImportError:
    grpc_available = False
    print("РЈР’РђР“Рђ: gRPC РјРѕРґСѓР»С– РЅРµ Р·РЅР°Р№РґРµРЅРѕ. gRPC Р±РµРЅС‡РјР°СЂРєС–РЅРі Р±СѓРґРµ РЅРµРґРѕСЃС‚СѓРїРЅРёР№.")

class BenchmarkResult:
    """
    РљР»Р°СЃ РґР»СЏ Р·Р±РµСЂС–РіР°РЅРЅСЏ СЂРµР·СѓР»СЊС‚Р°С‚С–РІ Р±РµРЅС‡РјР°СЂРєС–РЅРіСѓ
    """
    def __init__(self, name, protocol):
        """
        Р†РЅС–С†С–Р°Р»С–Р·Р°С†С–СЏ СЂРµР·СѓР»СЊС‚Р°С‚С–РІ Р±РµРЅС‡РјР°СЂРєС–РЅРіСѓ

        РџР°СЂР°РјРµС‚СЂРё:
        -----------
        name: РЅР°Р·РІР° С‚РµСЃС‚Сѓ
        protocol: РїСЂРѕС‚РѕРєРѕР» (REST Р°Р±Рѕ gRPC)
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
        РџРѕС‡Р°С‚РѕРє Р±РµРЅС‡РјР°СЂРєС–РЅРіСѓ
        """
        self.start_time = time.time()

    def end(self):
        """
        Р—Р°РІРµСЂС€РµРЅРЅСЏ Р±РµРЅС‡РјР°СЂРєС–РЅРіСѓ
        """
        self.end_time = time.time()
        self.total_time = self.end_time - self.start_time
        if self.total_time > 0:
            self.rps = self.successful_requests / self.total_time

    def add_result(self, success, latency, server_time=None):
        """
        Р”РѕРґР°РІР°РЅРЅСЏ СЂРµР·СѓР»СЊС‚Р°С‚Сѓ Р·Р°РїРёС‚Сѓ

        РџР°СЂР°РјРµС‚СЂРё:
        -----------
        success: СѓСЃРїС–С€РЅС–СЃС‚СЊ Р·Р°РїРёС‚Сѓ
        latency: С‡Р°СЃ РІРёРєРѕРЅР°РЅРЅСЏ Р·Р°РїРёС‚Сѓ
        server_time: С‡Р°СЃ РѕР±СЂРѕР±РєРё РЅР° СЃРµСЂРІРµСЂС–
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
        Р”РѕРґР°РІР°РЅРЅСЏ РїРѕРјРёР»РєРё

        РџР°СЂР°РјРµС‚СЂРё:
        -----------
        error: С‚РµРєСЃС‚ РїРѕРјРёР»РєРё
        """
        self.errors.append(str(error))

    def get_statistics(self):
        """
        РћР±С‡РёСЃР»РµРЅРЅСЏ СЃС‚Р°С‚РёСЃС‚РёРєРё

        РџРѕРІРµСЂС‚Р°С”:
        -----------
        СЃР»РѕРІРЅРёРє Р·С– СЃС‚Р°С‚РёСЃС‚РёРєРѕСЋ
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
                    'min': min(self.latencies) * 1000,  # РІ РјСЃ
                    'max': max(self.latencies) * 1000,  # РІ РјСЃ
                    'mean': statistics.mean(self.latencies) * 1000,  # РІ РјСЃ
                    'median': statistics.median(self.latencies) * 1000,  # РІ РјСЃ
                    'p90': np.percentile(self.latencies, 90) * 1000,  # РІ РјСЃ
                    'p95': np.percentile(self.latencies, 95) * 1000,  # РІ РјСЃ
                    'p99': np.percentile(self.latencies, 99) * 1000   # РІ РјСЃ
                }
            })

        if self.server_times:
            stats.update({
                'server_time': {
                    'min': min(self.server_times) * 1000,  # РІ РјСЃ
                    'max': max(self.server_times) * 1000,  # РІ РјСЃ
                    'mean': statistics.mean(self.server_times) * 1000,  # РІ РјСЃ
                    'median': statistics.median(self.server_times) * 1000,  # РІ РјСЃ
                    'p90': np.percentile(self.server_times, 90) * 1000,  # РІ РјСЃ
                    'p95': np.percentile(self.server_times, 95) * 1000,  # РІ РјСЃ
                    'p99': np.percentile(self.server_times, 99) * 1000   # РІ РјСЃ
                }
            })

        if self.network_times:
            stats.update({
                'network_time': {
                    'min': min(self.network_times) * 1000,  # РІ РјСЃ
                    'max': max(self.network_times) * 1000,  # РІ РјСЃ
                    'mean': statistics.mean(self.network_times) * 1000,  # РІ РјСЃ
                    'median': statistics.median(self.network_times) * 1000,  # РІ РјСЃ
                    'p90': np.percentile(self.network_times, 90) * 1000,  # РІ РјСЃ
                    'p95': np.percentile(self.network_times, 95) * 1000,  # РІ РјСЃ
                    'p99': np.percentile(self.network_times, 99) * 1000   # РІ РјСЃ
                }
            })

        if self.errors:
            stats['errors'] = self.errors[:10]  # РѕР±РјРµР¶СѓС”РјРѕ РєС–Р»СЊРєС–СЃС‚СЊ РїРѕРјРёР»РѕРє
            stats['error_count'] = len(self.errors)

        return stats

class RestClient:
    """
    РљР»С–С”РЅС‚ РґР»СЏ Р±РµРЅС‡РјР°СЂРєС–РЅРіСѓ REST API
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
        self.timeout = timeout

    def predict(self, image_path):
        """
        Р’С–РґРїСЂР°РІР»СЏС” Р·Р°РїРёС‚ РЅР° РїСЂРѕРіРЅРѕР·СѓРІР°РЅРЅСЏ

        РџР°СЂР°РјРµС‚СЂРё:
        -----------
        image_path: С€Р»СЏС… РґРѕ С„Р°Р№Р»Сѓ Р·РѕР±СЂР°Р¶РµРЅРЅСЏ

        РџРѕРІРµСЂС‚Р°С”:
        -----------
        РєРѕСЂС‚РµР¶ (СѓСЃРїС–С…, С‡Р°СЃ РІРёРєРѕРЅР°РЅРЅСЏ, С‡Р°СЃ СЃРµСЂРІРµСЂР°)
        """
        try:
            with open(image_path, 'rb') as f:
                files = {'file': (os.path.basename(image_path), f, 'image/jpeg')}

                start_time = time.time()
                response = requests.post(f"{self.base_url}/predict", files=files, timeout=self.timeout)
                elapsed = time.time() - start_time

                if response.status_code == 200:
                    result = response.json()
                    # РЎРїСЂРѕР±СѓС”РјРѕ РѕС‚СЂРёРјР°С‚Рё С‡Р°СЃ РѕР±СЂРѕР±РєРё РЅР° СЃРµСЂРІРµСЂС–, СЏРєС‰Рѕ РґРѕСЃС‚СѓРїРЅРѕ
                    server_time = result.get('processing_time', None)
                    if server_time is not None:
                        server_time = server_time / 1000  # РєРѕРЅРІРµСЂС‚СѓС”РјРѕ Р· РјСЃ Сѓ СЃРµРєСѓРЅРґРё
                    return True, elapsed, server_time
                else:
                    return False, elapsed, None
        except Exception as e:
            return False, 0, None

class GrpcClient:
    """
    РљР»С–С”РЅС‚ РґР»СЏ Р±РµРЅС‡РјР°СЂРєС–РЅРіСѓ gRPC API
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
            raise ImportError("gRPC РјРѕРґСѓР»С– РЅРµ РґРѕСЃС‚СѓРїРЅС–")

        # РЎС‚РІРѕСЂРµРЅРЅСЏ РєР°РЅР°Р»Сѓ Р· РѕРїС†С–СЏРјРё РґР»СЏ РІРµР»РёРєРёС… РїРѕРІС–РґРѕРјР»РµРЅСЊ
        channel_options = [
            ('grpc.max_send_message_length', 50 * 1024 * 1024),  # 50 MB
            ('grpc.max_receive_message_length', 50 * 1024 * 1024),  # 50 MB
        ]
        self.channel = grpc.insecure_channel(server_address, options=channel_options)
        self.stub = inference_pb2_grpc.InferenceServiceStub(self.channel)
        self.timeout = timeout

    def predict(self, image_path):
        """
        Р’С–РґРїСЂР°РІР»СЏС” Р·Р°РїРёС‚ РЅР° РїСЂРѕРіРЅРѕР·СѓРІР°РЅРЅСЏ

        РџР°СЂР°РјРµС‚СЂРё:
        -----------
        image_path: С€Р»СЏС… РґРѕ С„Р°Р№Р»Сѓ Р·РѕР±СЂР°Р¶РµРЅРЅСЏ

        РџРѕРІРµСЂС‚Р°С”:
        -----------
        РєРѕСЂС‚РµР¶ (СѓСЃРїС–С…, С‡Р°СЃ РІРёРєРѕРЅР°РЅРЅСЏ, С‡Р°СЃ СЃРµСЂРІРµСЂР°)
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

            if response.success:
                server_time = response.processing_time / 1000  # РєРѕРЅРІРµСЂС‚СѓС”РјРѕ Р· РјСЃ Сѓ СЃРµРєСѓРЅРґРё
                return True, elapsed, server_time
            else:
                return False, elapsed, None

        except Exception as e:
            return False, 0, None

    def close(self):
        """
        Р—Р°РєСЂРёС‚С‚СЏ Р·'С”РґРЅР°РЅРЅСЏ
        """
        self.channel.close()

def run_benchmark(client, image_path, num_requests, concurrency, result):
    """
    Р—Р°РїСѓСЃРєР°С” Р±РµРЅС‡РјР°СЂРєС–РЅРі

    РџР°СЂР°РјРµС‚СЂРё:
    -----------
    client: РєР»С–С”РЅС‚ (RestClient Р°Р±Рѕ GrpcClient)
    image_path: С€Р»СЏС… РґРѕ С„Р°Р№Р»Сѓ Р·РѕР±СЂР°Р¶РµРЅРЅСЏ
    num_requests: РєС–Р»СЊРєС–СЃС‚СЊ Р·Р°РїРёС‚С–РІ
    concurrency: СЂС–РІРµРЅСЊ РїР°СЂР°Р»РµР»С–Р·РјСѓ
    result: РѕР±'С”РєС‚ BenchmarkResult РґР»СЏ Р·Р±РµСЂРµР¶РµРЅРЅСЏ СЂРµР·СѓР»СЊС‚Р°С‚С–РІ

    РџРѕРІРµСЂС‚Р°С”:
    -----------
    РѕР±'С”РєС‚ BenchmarkResult Р· СЂРµР·СѓР»СЊС‚Р°С‚Р°РјРё
    """
    result.concurrency = concurrency
    result.start()

    def send_request():
        try:
            success, latency, server_time = client.predict(image_path)
            result.add_result(success, latency, server_time)
            if not success:
                result.add_error("Р—Р°РїРёС‚ РЅРµРІРґР°Р»РёР№")
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
    Р’РёРІРѕРґРёС‚СЊ СЂРµР·СѓР»СЊС‚Р°С‚Рё Р±РµРЅС‡РјР°СЂРєС–РЅРіСѓ

    РџР°СЂР°РјРµС‚СЂРё:
    -----------
    result: РѕР±'С”РєС‚ BenchmarkResult
    """
    stats = result.get_statistics()

    print(f"\nР РµР·СѓР»СЊС‚Р°С‚Рё Р±РµРЅС‡РјР°СЂРєС–РЅРіСѓ РґР»СЏ {stats['name']} ({stats['protocol']}):\n")
    print(f"Р—Р°РіР°Р»СЊРЅРёР№ С‡Р°СЃ: {stats['total_time']:.2f} СЃ")
    print(f"Р—Р°РїРёС‚С–РІ: {stats['total_requests']}")
    print(f"РЈСЃРїС–С€РЅРёС…: {stats['successful_requests']} ({stats['success_rate']:.2f}%)")
    print(f"РќРµРІРґР°Р»РёС…: {stats['failed_requests']}")
    print(f"Р—Р°РїРёС‚С–РІ РЅР° СЃРµРєСѓРЅРґСѓ (RPS): {stats['requests_per_second']:.2f}")
    print(f"Р С–РІРµРЅСЊ РїР°СЂР°Р»РµР»С–Р·РјСѓ: {stats['concurrency']}")

    if 'latency' in stats:
        print("\nР§Р°СЃ РІРёРєРѕРЅР°РЅРЅСЏ Р·Р°РїРёС‚Сѓ (РјСЃ):")
        print(f"  РњС–РЅ: {stats['latency']['min']:.2f}")
        print(f"  РњР°РєСЃ: {stats['latency']['max']:.2f}")
        print(f"  РЎРµСЂРµРґРЅС”: {stats['latency']['mean']:.2f}")
        print(f"  РњРµРґС–Р°РЅР°: {stats['latency']['median']:.2f}")
        print(f"  P90: {stats['latency']['p90']:.2f}")
        print(f"  P95: {stats['latency']['p95']:.2f}")
        print(f"  P99: {stats['latency']['p99']:.2f}")

    if 'server_time' in stats:
        print("\nР§Р°СЃ РѕР±СЂРѕР±РєРё РЅР° СЃРµСЂРІРµСЂС– (РјСЃ):")
        print(f"  РњС–РЅ: {stats['server_time']['min']:.2f}")
        print(f"  РњР°РєСЃ: {stats['server_time']['max']:.2f}")
        print(f"  РЎРµСЂРµРґРЅС”: {stats['server_time']['mean']:.2f}")
        print(f"  РњРµРґС–Р°РЅР°: {stats['server_time']['median']:.2f}")
        print(f"  P90: {stats['server_time']['p90']:.2f}")
        print(f"  P95: {stats['server_time']['p95']:.2f}")
        print(f"  P99: {stats['server_time']['p99']:.2f}")

    if 'network_time' in stats:
        print("\nРњРµСЂРµР¶РµРІР° Р·Р°С‚СЂРёРјРєР° (РјСЃ):")
        print(f"  РњС–РЅ: {stats['network_time']['min']:.2f}")
        print(f"  РњР°РєСЃ: {stats['network_time']['max']:.2f}")
        print(f"  РЎРµСЂРµРґРЅС”: {stats['network_time']['mean']:.2f}")
        print(f"  РњРµРґС–Р°РЅР°: {stats['network_time']['median']:.2f}")
        print(f"  P90: {stats['network_time']['p90']:.2f}")
        print(f"  P95: {stats['network_time']['p95']:.2f}")
        print(f"  P99: {stats['network_time']['p99']:.2f}")

    if 'errors' in stats and stats['errors']:
        print(f"\nРџРѕРјРёР»РєРё ({stats['error_count']} РІСЃСЊРѕРіРѕ):")
        for i, error in enumerate(stats['errors']):
            print(f"  {i+1}. {error}")

def save_results_csv(results, output_file):
    """
    Р—Р±РµСЂС–РіР°С” СЂРµР·СѓР»СЊС‚Р°С‚Рё Сѓ CSV С„Р°Р№Р»

    РџР°СЂР°РјРµС‚СЂРё:
    -----------
    results: СЃРїРёСЃРѕРє РѕР±'С”РєС‚С–РІ BenchmarkResult
    output_file: С€Р»СЏС… РґРѕ РІРёС…С–РґРЅРѕРіРѕ С„Р°Р№Р»Сѓ
    """
    stats_list = [result.get_statistics() for result in results]

    # Р’РёР·РЅР°С‡РµРЅРЅСЏ Р·Р°РіРѕР»РѕРІРєС–РІ
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

        # Р”РѕРґР°РІР°РЅРЅСЏ РјРµС‚СЂРёРє Р·Р°С‚СЂРёРјРєРё
        if 'latency' in stats:
            for key in ['min', 'max', 'mean', 'median', 'p90', 'p95', 'p99']:
                row[f'latency_{key}'] = stats['latency'][key]
        else:
            for key in ['min', 'max', 'mean', 'median', 'p90', 'p95', 'p99']:
                row[f'latency_{key}'] = None

        # Р”РѕРґР°РІР°РЅРЅСЏ РјРµС‚СЂРёРє С‡Р°СЃСѓ СЃРµСЂРІРµСЂР°
        if 'server_time' in stats:
            for key in ['min', 'max', 'mean', 'median', 'p90', 'p95', 'p99']:
                row[f'server_time_{key}'] = stats['server_time'][key]
        else:
            for key in ['min', 'max', 'mean', 'median', 'p90', 'p95', 'p99']:
                row[f'server_time_{key}'] = None

        # Р”РѕРґР°РІР°РЅРЅСЏ РјРµС‚СЂРёРє РјРµСЂРµР¶РµРІРѕС— Р·Р°С‚СЂРёРјРєРё
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

    print(f"Р РµР·СѓР»СЊС‚Р°С‚Рё Р·Р±РµСЂРµР¶РµРЅРѕ Сѓ {output_file}")

def save_results_json(results, output_file):
    """
    Р—Р±РµСЂС–РіР°С” СЂРµР·СѓР»СЊС‚Р°С‚Рё Сѓ JSON С„Р°Р№Р»

    РџР°СЂР°РјРµС‚СЂРё:
    -----------
    results: СЃРїРёСЃРѕРє РѕР±'С”РєС‚С–РІ BenchmarkResult
    output_file: С€Р»СЏС… РґРѕ РІРёС…С–РґРЅРѕРіРѕ С„Р°Р№Р»Сѓ
    """
    stats_list = [result.get_statistics() for result in results]

    with open(output_file, 'w') as f:
        json.dump(stats_list, f, indent=2)

    print(f"Р РµР·СѓР»СЊС‚Р°С‚Рё Р·Р±РµСЂРµР¶РµРЅРѕ Сѓ {output_file}")

def plot_results(results, output_file=None):
    """
    РЎС‚РІРѕСЂСЋС” РІС–Р·СѓР°Р»С–Р·Р°С†С–СЋ СЂРµР·СѓР»СЊС‚Р°С‚С–РІ

    РџР°СЂР°РјРµС‚СЂРё:
    -----------
    results: СЃРїРёСЃРѕРє РѕР±'С”РєС‚С–РІ BenchmarkResult
    output_file: С€Р»СЏС… РґРѕ РІРёС…С–РґРЅРѕРіРѕ С„Р°Р№Р»Сѓ (СЏРєС‰Рѕ None, РіСЂР°С„С–РєРё РІС–РґРѕР±СЂР°Р¶Р°СЋС‚СЊСЃСЏ)
    """
    stats_list = [result.get_statistics() for result in results]

    # РЎС‚РІРѕСЂРµРЅРЅСЏ РіСЂР°С„С–РєС–РІ
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Р РµР·СѓР»СЊС‚Р°С‚Рё Р±РµРЅС‡РјР°СЂРєС–РЅРіСѓ', fontsize=16)

    # Р“СЂР°С„С–Рє 1: RPS
    names = [f"{stats['name']}\n({stats['protocol']})" for stats in stats_list]
    rps = [stats['requests_per_second'] for stats in stats_list]

    axes[0, 0].bar(names, rps)
    axes[0, 0].set_title('Р—Р°РїРёС‚Рё РЅР° СЃРµРєСѓРЅРґСѓ (RPS)')
    axes[0, 0].set_ylabel('RPS')
    axes[0, 0].grid(axis='y', linestyle='--', alpha=0.7)

    # Р“СЂР°С„С–Рє 2: Latency
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

    axes[0, 1].set_title('Р§Р°СЃ РІРёРєРѕРЅР°РЅРЅСЏ Р·Р°РїРёС‚Сѓ (РјСЃ)')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(names)
    axes[0, 1].set_ylabel('Р§Р°СЃ (РјСЃ)')
    axes[0, 1].legend()
    axes[0, 1].grid(axis='y', linestyle='--', alpha=0.7)

    # Р“СЂР°С„С–Рє 3: РџРѕСЂС–РІРЅСЏРЅРЅСЏ С‡Р°СЃСѓ СЃРµСЂРІРµСЂР° С– РјРµСЂРµР¶С–
    server_times = []
    network_times = []

    for stats in stats_list:
        if 'server_time' in stats and 'network_time' in stats:
            server_times.append(stats['server_time']['mean'])
            network_times.append(stats['network_time']['mean'])
        else:
            server_times.append(0)
            network_times.append(0)

    axes[1, 0].bar(names, server_times, label='Р§Р°СЃ СЃРµСЂРІРµСЂР°')
    axes[1, 0].bar(names, network_times, bottom=server_times, label='РњРµСЂРµР¶РµРІР° Р·Р°С‚СЂРёРјРєР°')

    axes[1, 0].set_title('Р РѕР·РїРѕРґС–Р» С‡Р°СЃСѓ РІРёРєРѕРЅР°РЅРЅСЏ Р·Р°РїРёС‚Сѓ (РјСЃ)')
    axes[1, 0].set_ylabel('Р§Р°СЃ (РјСЃ)')
    axes[1, 0].legend()
    axes[1, 0].grid(axis='y', linestyle='--', alpha=0.7)

    # Р“СЂР°С„С–Рє 4: РЈСЃРїС–С€РЅС–СЃС‚СЊ Р·Р°РїРёС‚С–РІ
    success_rates = [stats['success_rate'] for stats in stats_list]

    axes[1, 1].bar(names, success_rates)
    axes[1, 1].set_title('Р’С–РґСЃРѕС‚РѕРє СѓСЃРїС–С€РЅРёС… Р·Р°РїРёС‚С–РІ (%)')
    axes[1, 1].set_ylabel('Р’С–РґСЃРѕС‚РѕРє (%)')
    axes[1, 1].set_ylim(0, 100)
    axes[1, 1].grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if output_file:
        plt.savefig(output_file)
        print(f"Р“СЂР°С„С–РєРё Р·Р±РµСЂРµР¶РµРЅРѕ Сѓ {output_file}")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='Р†РЅСЃС‚СЂСѓРјРµРЅС‚ РґР»СЏ Р±РµРЅС‡РјР°СЂРєС–РЅРіСѓ СЃРµСЂРІРµСЂС–РІ РјРѕРґРµР»РµР№')

    # РћСЃРЅРѕРІРЅС– РїР°СЂР°РјРµС‚СЂРё
    parser.add_argument('--mode', type=str, choices=['rest', 'grpc', 'both'], default='both',
                        help='Р РµР¶РёРј Р±РµРЅС‡РјР°СЂРєС–РЅРіСѓ: REST, gRPC Р°Р±Рѕ РѕР±РёРґРІР°')
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
    parser.add_argument('--output-csv', type=str, default=None,
                        help='РЁР»СЏС… РґР»СЏ Р·Р±РµСЂРµР¶РµРЅРЅСЏ СЂРµР·СѓР»СЊС‚Р°С‚С–РІ Сѓ CSV')
    parser.add_argument('--output-json', type=str, default=None,
                        help='РЁР»СЏС… РґР»СЏ Р·Р±РµСЂРµР¶РµРЅРЅСЏ СЂРµР·СѓР»СЊС‚Р°С‚С–РІ Сѓ JSON')
    parser.add_argument('--output-plot', type=str, default=None,
                        help='РЁР»СЏС… РґР»СЏ Р·Р±РµСЂРµР¶РµРЅРЅСЏ РіСЂР°С„С–РєС–РІ')

    # РџР°СЂР°РјРµС‚СЂРё РґР»СЏ РїРѕСЂС–РІРЅСЏРЅРЅСЏ СЂС–Р·РЅРёС… РєРѕРЅС„С–РіСѓСЂР°С†С–Р№
    parser.add_argument('--concurrency-range', type=str, default=None,
                        help='Р”С–Р°РїР°Р·РѕРЅ СЂС–РІРЅС–РІ РїР°СЂР°Р»РµР»С–Р·РјСѓ Сѓ С„РѕСЂРјР°С‚С– "start,end,step"')

    args = parser.parse_args()

    # РџРµСЂРµРІС–СЂРєР° РЅР°СЏРІРЅРѕСЃС‚С– С„Р°Р№Р»Сѓ Р·РѕР±СЂР°Р¶РµРЅРЅСЏ
    if not os.path.isfile(args.image):
        print(f"РџРѕРјРёР»РєР°: С„Р°Р№Р» {args.image} РЅРµ С–СЃРЅСѓС”")
        return 1

    # РџС–РґРіРѕС‚РѕРІРєР° СЃРїРёСЃРєСѓ СЂС–РІРЅС–РІ РїР°СЂР°Р»РµР»С–Р·РјСѓ РґР»СЏ С‚РµСЃС‚СѓРІР°РЅРЅСЏ
    concurrency_levels = [args.concurrency]
    if args.concurrency_range:
        try:
            start, end, step = map(int, args.concurrency_range.split(','))
            concurrency_levels = list(range(start, end + 1, step))
        except ValueError:
            print(f"РџРѕРјРёР»РєР°: РЅРµРІС–СЂРЅРёР№ С„РѕСЂРјР°С‚ РґС–Р°РїР°Р·РѕРЅСѓ РїР°СЂР°Р»РµР»С–Р·РјСѓ: {args.concurrency_range}")
            return 1

    # Р—Р°РїСѓСЃРє Р±РµРЅС‡РјР°СЂРєС–РЅРіСѓ
    results = []

    # REST API Р±РµРЅС‡РјР°СЂРєС–РЅРі
    if args.mode in ['rest', 'both']:
        rest_client = RestClient(args.rest_url, timeout=args.timeout)

        for concurrency in concurrency_levels:
            print(f"\nР—Р°РїСѓСЃРє REST Р±РµРЅС‡РјР°СЂРєС–РЅРіСѓ Р· {args.requests} Р·Р°РїРёС‚Р°РјРё С‚Р° СЂС–РІРЅРµРј РїР°СЂР°Р»РµР»С–Р·РјСѓ {concurrency}...")
            result = BenchmarkResult(f"REST-C{concurrency}", "REST")
            run_benchmark(rest_client, args.image, args.requests, concurrency, result)
            print_results(result)
            results.append(result)

    # gRPC Р±РµРЅС‡РјР°СЂРєС–РЅРі
    if args.mode in ['grpc', 'both'] and grpc_available:
        try:
            grpc_client = GrpcClient(args.grpc_server, timeout=args.timeout)

            for concurrency in concurrency_levels:
                print(f"\nР—Р°РїСѓСЃРє gRPC Р±РµРЅС‡РјР°СЂРєС–РЅРіСѓ Р· {args.requests} Р·Р°РїРёС‚Р°РјРё С‚Р° СЂС–РІРЅРµРј РїР°СЂР°Р»РµР»С–Р·РјСѓ {concurrency}...")
                result = BenchmarkResult(f"gRPC-C{concurrency}", "gRPC")
                run_benchmark(grpc_client, args.image, args.requests, concurrency, result)
                print_results(result)
                results.append(result)

            grpc_client.close()
        except Exception as e:
            print(f"РџРѕРјРёР»РєР° РїСЂРё gRPC Р±РµРЅС‡РјР°СЂРєС–РЅРіСѓ: {e}")

    # Р—Р±РµСЂРµР¶РµРЅРЅСЏ СЂРµР·СѓР»СЊС‚Р°С‚С–РІ
    if args.output_csv:
        save_results_csv(results, args.output_csv)

    if args.output_json:
        save_results_json(results, args.output_json)

    if args.output_plot or len(results) > 1:
        plot_results(results, args.output_plot)

    return 0

if __name__ == "__main__":
    sys.exit(main())

