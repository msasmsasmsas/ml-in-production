# Updated version for PR
#!/usr/bin/env python
"""
РЎРєСЂРёРїС‚ РґР»СЏ РїРѕСЂС–РІРЅСЏР»СЊРЅРѕРіРѕ Р±РµРЅС‡РјР°СЂРєС–РЅРіСѓ REST С‚Р° gRPC С–РЅС‚РµСЂС„РµР№СЃС–РІ РјРѕРґРµР»РµР№ РјР°С€РёРЅРЅРѕРіРѕ РЅР°РІС‡Р°РЅРЅСЏ
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
    print("РЈР’РђР“Рђ: gRPC РјРѕРґСѓР»С– РЅРµ Р·РЅР°Р№РґРµРЅРѕ. gRPC Р±РµРЅС‡РјР°СЂРєС–РЅРі Р±СѓРґРµ РЅРµРґРѕСЃС‚СѓРїРЅРёР№.")

class RestClient:
    """
    РљР»С–С”РЅС‚ РґР»СЏ REST API
    """
    def __init__(self, base_url):
        """
        Р†РЅС–С†С–Р°Р»С–Р·Р°С†С–СЏ РєР»С–С”РЅС‚Р°

        РџР°СЂР°РјРµС‚СЂРё:
        -----------
        base_url: Р±Р°Р·РѕРІР° URL Р°РґСЂРµСЃР° СЃРµСЂРІРµСЂР°
        """
        self.base_url = base_url
        self.predict_url = f"{base_url}/predict"

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
                response = requests.post(self.predict_url, files=files)
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
    def __init__(self, server_address):
        """
        Р†РЅС–С†С–Р°Р»С–Р·Р°С†С–СЏ РєР»С–С”РЅС‚Р°

        РџР°СЂР°РјРµС‚СЂРё:
        -----------
        server_address: Р°РґСЂРµСЃР° gRPC СЃРµСЂРІРµСЂР°
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
            response = self.stub.Predict(request)
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

def run_benchmark(client, image_path, num_requests, concurrency):
    """
    Р—Р°РїСѓСЃРєР°С” Р±РµРЅС‡РјР°СЂРєС–РЅРі

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

    # РћР±С‡РёСЃР»РµРЅРЅСЏ СЃС‚Р°С‚РёСЃС‚РёРєРё
    latencies = [r['elapsed'] for r in results if r['success']]

    if not latencies:
        print("Р’СЃС– Р·Р°РїРёС‚Рё Р·Р°РІРµСЂС€РёР»РёСЃСЏ Р· РїРѕРјРёР»РєР°РјРё")
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
    Р’РёРІРѕРґРёС‚СЊ СЂРµР·СѓР»СЊС‚Р°С‚Рё Р±РµРЅС‡РјР°СЂРєС–РЅРіСѓ

    РџР°СЂР°РјРµС‚СЂРё:
    -----------
    protocol: РЅР°Р·РІР° РїСЂРѕС‚РѕРєРѕР»Сѓ (REST Р°Р±Рѕ gRPC)
    results: СЃР»РѕРІРЅРёРє Р· СЂРµР·СѓР»СЊС‚Р°С‚Р°РјРё
    """
    print(f"\nР РµР·СѓР»СЊС‚Р°С‚Рё Р±РµРЅС‡РјР°СЂРєС–РЅРіСѓ {protocol}:")
    print(f"Р—Р°РіР°Р»СЊРЅРёР№ С‡Р°СЃ: {results['total_time']:.2f} СЃ")
    print(f"Р—Р°РїРёС‚С–РІ: {results['total_requests']}")
    print(f"РЈСЃРїС–С€РЅРёС…: {results['successful_requests']} ({100 * results['successful_requests'] / results['total_requests']:.2f}%)")
    print(f"РќРµРІРґР°Р»РёС…: {results['failed_requests']}")
    print(f"Р С–РІРµРЅСЊ РїР°СЂР°Р»РµР»С–Р·РјСѓ: {results['concurrency']}")

    if results['stats']:
        stats = results['stats']
        print("\nРЎС‚Р°С‚РёСЃС‚РёРєР° С‡Р°СЃСѓ РІРёРєРѕРЅР°РЅРЅСЏ (РјСЃ):")
        print(f"  РњС–РЅ: {stats['min']:.2f}")
        print(f"  РњР°РєСЃ: {stats['max']:.2f}")
        print(f"  РЎРµСЂРµРґРЅС”: {stats['mean']:.2f}")
        print(f"  РњРµРґС–Р°РЅР°: {stats['median']:.2f}")
        print(f"  P90: {stats['p90']:.2f}")
        print(f"  P95: {stats['p95']:.2f}")
        print(f"  P99: {stats['p99']:.2f}")
        print(f"  РЎС‚Р°РЅРґР°СЂС‚РЅРµ РІС–РґС…РёР»РµРЅРЅСЏ: {stats['std']:.2f}")
        print(f"  Р—Р°РїРёС‚С–РІ РЅР° СЃРµРєСѓРЅРґСѓ (RPS): {stats['rps']:.2f}")

def compare_and_plot(rest_results, grpc_results, output_file=None):
    """
    РџРѕСЂС–РІРЅСЋС” С‚Р° РІС–Р·СѓР°Р»С–Р·СѓС” СЂРµР·СѓР»СЊС‚Р°С‚Рё Р±РµРЅС‡РјР°СЂРєС–РЅРіСѓ REST С‚Р° gRPC

    РџР°СЂР°РјРµС‚СЂРё:
    -----------
    rest_results: СЃР»РѕРІРЅРёРє Р· СЂРµР·СѓР»СЊС‚Р°С‚Р°РјРё REST
    grpc_results: СЃР»РѕРІРЅРёРє Р· СЂРµР·СѓР»СЊС‚Р°С‚Р°РјРё gRPC
    output_file: С€Р»СЏС… РґРѕ РІРёС…С–РґРЅРѕРіРѕ С„Р°Р№Р»Сѓ (СЏРєС‰Рѕ None, РіСЂР°С„С–РєРё РІС–РґРѕР±СЂР°Р¶Р°СЋС‚СЊСЃСЏ)
    """
    if not rest_results['stats'] or not grpc_results['stats']:
        print("РќРµРґРѕСЃС‚Р°С‚РЅСЊРѕ РґР°РЅРёС… РґР»СЏ РїРѕСЂС–РІРЅСЏРЅРЅСЏ")
        return

    # РЎС‚РІРѕСЂРµРЅРЅСЏ РґР°С‚Р°С„СЂРµР№РјСѓ РґР»СЏ РІС–Р·СѓР°Р»С–Р·Р°С†С–С—
    rest_latencies = [l * 1000 for l in rest_results['raw_latencies']]  # РјСЃ
    grpc_latencies = [l * 1000 for l in grpc_results['raw_latencies']]  # РјСЃ

    rest_df = pd.DataFrame({'latency': rest_latencies, 'protocol': 'REST'})
    grpc_df = pd.DataFrame({'latency': grpc_latencies, 'protocol': 'gRPC'})
    df = pd.concat([rest_df, grpc_df])

    # РЎС‚РІРѕСЂРµРЅРЅСЏ РіСЂР°С„С–РєС–РІ
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('РџРѕСЂС–РІРЅСЏРЅРЅСЏ REST С‚Р° gRPC', fontsize=16)

    # Р“СЂР°С„С–Рє 1: RPS
    protocols = ['REST', 'gRPC']
    rps = [rest_results['stats']['rps'], grpc_results['stats']['rps']]

    axes[0, 0].bar(protocols, rps)
    axes[0, 0].set_title('Р—Р°РїРёС‚Рё РЅР° СЃРµРєСѓРЅРґСѓ (RPS)')
    axes[0, 0].set_ylabel('RPS')
    for i, v in enumerate(rps):
        axes[0, 0].text(i, v, f"{v:.2f}", ha='center', va='bottom')

    # Р“СЂР°С„С–Рє 2: Р РѕР·РїРѕРґС–Р» Р·Р°С‚СЂРёРјРѕРє
    sns.boxplot(x='protocol', y='latency', data=df, ax=axes[0, 1])
    axes[0, 1].set_title('Р РѕР·РїРѕРґС–Р» Р·Р°С‚СЂРёРјРѕРє')
    axes[0, 1].set_ylabel('Р—Р°С‚СЂРёРјРєР° (РјСЃ)')

    # Р“СЂР°С„С–Рє 3: Р“С–СЃС‚РѕРіСЂР°РјР° Р·Р°С‚СЂРёРјРѕРє
    sns.histplot(data=df, x='latency', hue='protocol', kde=True, ax=axes[1, 0])
    axes[1, 0].set_title('Р“С–СЃС‚РѕРіСЂР°РјР° Р·Р°С‚СЂРёРјРѕРє')
    axes[1, 0].set_xlabel('Р—Р°С‚СЂРёРјРєР° (РјСЃ)')
    axes[1, 0].set_ylabel('РљС–Р»СЊРєС–СЃС‚СЊ Р·Р°РїРёС‚С–РІ')

    # Р“СЂР°С„С–Рє 4: РџРѕСЂС–РІРЅСЏРЅРЅСЏ РјРµС‚СЂРёРє
    metrics = ['mean', 'median', 'p90', 'p95', 'p99']
    rest_metrics = [rest_results['stats'][m] for m in metrics]
    grpc_metrics = [grpc_results['stats'][m] for m in metrics]

    x = np.arange(len(metrics))
    width = 0.35

    axes[1, 1].bar(x - width/2, rest_metrics, width, label='REST')
    axes[1, 1].bar(x + width/2, grpc_metrics, width, label='gRPC')

    axes[1, 1].set_title('РџРѕСЂС–РІРЅСЏРЅРЅСЏ РјРµС‚СЂРёРє')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(metrics)
    axes[1, 1].set_ylabel('Р—Р°С‚СЂРёРјРєР° (РјСЃ)')
    axes[1, 1].legend()

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if output_file:
        plt.savefig(output_file)
        print(f"Р“СЂР°С„С–РєРё Р·Р±РµСЂРµР¶РµРЅРѕ Сѓ {output_file}")
    else:
        plt.show()

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

    print(f"Р РµР·СѓР»СЊС‚Р°С‚Рё Р·Р±РµСЂРµР¶РµРЅРѕ Сѓ {output_file}")

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

    # РџР°СЂР°РјРµС‚СЂРё РІРёС…С–РґРЅРёС… РґР°РЅРёС…
    parser.add_argument('--output-json', type=str, default=None,
                        help='РЁР»СЏС… РґР»СЏ Р·Р±РµСЂРµР¶РµРЅРЅСЏ СЂРµР·СѓР»СЊС‚Р°С‚С–РІ Сѓ JSON')
    parser.add_argument('--output-plot', type=str, default=None,
                        help='РЁР»СЏС… РґР»СЏ Р·Р±РµСЂРµР¶РµРЅРЅСЏ РіСЂР°С„С–РєС–РІ')

    args = parser.parse_args()

    # РџРµСЂРµРІС–СЂРєР° РЅР°СЏРІРЅРѕСЃС‚С– С„Р°Р№Р»Сѓ Р·РѕР±СЂР°Р¶РµРЅРЅСЏ
    if not os.path.isfile(args.image):
        print(f"РџРѕРјРёР»РєР°: С„Р°Р№Р» {args.image} РЅРµ С–СЃРЅСѓС”")
        return 1

    # REST Р±РµРЅС‡РјР°СЂРєС–РЅРі
    print(f"\nР—Р°РїСѓСЃРє REST Р±РµРЅС‡РјР°СЂРєС–РЅРіСѓ ({args.requests} Р·Р°РїРёС‚С–РІ, РїР°СЂР°Р»РµР»С–Р·Рј {args.concurrency})...")
    rest_client = RestClient(args.rest_url)
    rest_results = run_benchmark(rest_client, args.image, args.requests, args.concurrency)
    print_benchmark_results("REST", rest_results)

    # gRPC Р±РµРЅС‡РјР°СЂРєС–РЅРі
    grpc_results = None
    if grpc_available:
        try:
            print(f"\nР—Р°РїСѓСЃРє gRPC Р±РµРЅС‡РјР°СЂРєС–РЅРіСѓ ({args.requests} Р·Р°РїРёС‚С–РІ, РїР°СЂР°Р»РµР»С–Р·Рј {args.concurrency})...")
            grpc_client = GrpcClient(args.grpc_server)
            grpc_results = run_benchmark(grpc_client, args.image, args.requests, args.concurrency)
            print_benchmark_results("gRPC", grpc_results)
            grpc_client.close()
        except Exception as e:
            print(f"РџРѕРјРёР»РєР° РїСЂРё gRPC Р±РµРЅС‡РјР°СЂРєС–РЅРіСѓ: {e}")
    else:
        print("\ngRPC Р±РµРЅС‡РјР°СЂРєС–РЅРі РЅРµРґРѕСЃС‚СѓРїРЅРёР№ (РјРѕРґСѓР»С– РЅРµ Р·РЅР°Р№РґРµРЅРѕ)")

    # РџРѕСЂС–РІРЅСЏРЅРЅСЏ С‚Р° РІС–Р·СѓР°Р»С–Р·Р°С†С–СЏ
    if grpc_results and grpc_results['stats'] and rest_results['stats']:
        compare_and_plot(rest_results, grpc_results, args.output_plot)

    # Р—Р±РµСЂРµР¶РµРЅРЅСЏ СЂРµР·СѓР»СЊС‚Р°С‚С–РІ
    if args.output_json:
        save_results_json(rest_results, grpc_results, args.output_json)

    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())

