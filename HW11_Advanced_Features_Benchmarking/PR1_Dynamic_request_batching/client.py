# Updated version for PR
import requests
import time
import sys
import concurrent.futures
import argparse
import os
from pathlib import Path

class ModelClient:
    """
    РљР»С–С”РЅС‚ РґР»СЏ С‚РµСЃС‚СѓРІР°РЅРЅСЏ СЃРµСЂРІРµСЂР° Р· РґРёРЅР°РјС–С‡РЅРёРј РїР°РєРµС‚СѓРІР°РЅРЅСЏРј
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
        СЃР»РѕРІРЅРёРє Р· СЂРµР·СѓР»СЊС‚Р°С‚Р°РјРё РїСЂРѕРіРЅРѕР·СѓРІР°РЅРЅСЏ Р°Р±Рѕ РїРѕРјРёР»РєРѕСЋ
        """
        try:
            with open(image_path, 'rb') as f:
                files = {'file': (os.path.basename(image_path), f, 'image/jpeg')}
                start_time = time.time()
                response = requests.post(self.predict_url, files=files)
                elapsed = time.time() - start_time

                if response.status_code == 200:
                    result = response.json()
                    result['latency'] = elapsed
                    return result
                else:
                    return {'error': f'HTTP РїРѕРјРёР»РєР°: {response.status_code}', 'response': response.text, 'latency': elapsed}
        except Exception as e:
            return {'error': str(e), 'latency': 0}

def run_concurrent_test(client, image_path, num_requests, concurrency):
    """
    Р—Р°РїСѓСЃРєР°С” РєРѕРЅРєСѓСЂРµРЅС‚РЅРµ С‚РµСЃС‚СѓРІР°РЅРЅСЏ СЃРµСЂРІРµСЂР°

    РџР°СЂР°РјРµС‚СЂРё:
    -----------
    client: РµРєР·РµРјРїР»СЏСЂ РєР»Р°СЃСѓ ModelClient
    image_path: С€Р»СЏС… РґРѕ С„Р°Р№Р»Сѓ Р·РѕР±СЂР°Р¶РµРЅРЅСЏ
    num_requests: Р·Р°РіР°Р»СЊРЅР° РєС–Р»СЊРєС–СЃС‚СЊ Р·Р°РїРёС‚С–РІ
    concurrency: РєС–Р»СЊРєС–СЃС‚СЊ РїР°СЂР°Р»РµР»СЊРЅРёС… Р·Р°РїРёС‚С–РІ
    """
    print(f"Р—Р°РїСѓСЃРє {num_requests} Р·Р°РїРёС‚С–РІ Р· СЂС–РІРЅРµРј РїР°СЂР°Р»РµР»С–Р·РјСѓ {concurrency}")

    results = []
    errors = 0

    start_time = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        future_to_id = {executor.submit(client.predict, image_path): i for i in range(num_requests)}

        for future in concurrent.futures.as_completed(future_to_id):
            result = future.result()
            if 'error' in result:
                errors += 1
            results.append(result)

    total_time = time.time() - start_time

    # РћР±С‡РёСЃР»РµРЅРЅСЏ СЃС‚Р°С‚РёСЃС‚РёРєРё
    latencies = [r['latency'] for r in results]
    avg_latency = sum(latencies) / len(latencies)
    min_latency = min(latencies)
    max_latency = max(latencies)
    p95_latency = sorted(latencies)[int(len(latencies) * 0.95)]
    rps = num_requests / total_time

    print(f"\nР РµР·СѓР»СЊС‚Р°С‚Рё С‚РµСЃС‚СѓРІР°РЅРЅСЏ:")
    print(f"Р—Р°РіР°Р»СЊРЅРёР№ С‡Р°СЃ: {total_time:.2f} СЃ")
    print(f"РЈСЃРїС–С€РЅРёС… Р·Р°РїРёС‚С–РІ: {num_requests - errors} Р· {num_requests} ({100 * (num_requests - errors) / num_requests:.2f}%)")
    print(f"RPS (Р·Р°РїРёС‚С–РІ РЅР° СЃРµРєСѓРЅРґСѓ): {rps:.2f}")
    print(f"РЎРµСЂРµРґРЅСЏ Р·Р°С‚СЂРёРјРєР°: {avg_latency * 1000:.2f} РјСЃ")
    print(f"РњС–РЅС–РјР°Р»СЊРЅР° Р·Р°С‚СЂРёРјРєР°: {min_latency * 1000:.2f} РјСЃ")
    print(f"РњР°РєСЃРёРјР°Р»СЊРЅР° Р·Р°С‚СЂРёРјРєР°: {max_latency * 1000:.2f} РјСЃ")
    print(f"P95 Р·Р°С‚СЂРёРјРєР°: {p95_latency * 1000:.2f} РјСЃ")

    return {
        'total_time': total_time,
        'successful_requests': num_requests - errors,
        'total_requests': num_requests,
        'rps': rps,
        'avg_latency_ms': avg_latency * 1000,
        'min_latency_ms': min_latency * 1000,
        'max_latency_ms': max_latency * 1000,
        'p95_latency_ms': p95_latency * 1000
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='РљР»С–С”РЅС‚ РґР»СЏ С‚РµСЃС‚СѓРІР°РЅРЅСЏ СЃРµСЂРІРµСЂР° Р· РґРёРЅР°РјС–С‡РЅРёРј РїР°РєРµС‚СѓРІР°РЅРЅСЏРј')
    parser.add_argument('--url', type=str, default='http://localhost:5000', help='Р‘Р°Р·РѕРІР° URL СЃРµСЂРІРµСЂР°')
    parser.add_argument('--image', type=str, default='test_image.jpg', help='РЁР»СЏС… РґРѕ С‚РµСЃС‚РѕРІРѕРіРѕ Р·РѕР±СЂР°Р¶РµРЅРЅСЏ')
    parser.add_argument('--requests', type=int, default=100, help='РљС–Р»СЊРєС–СЃС‚СЊ Р·Р°РїРёС‚С–РІ')
    parser.add_argument('--concurrency', type=int, default=10, help='Р С–РІРµРЅСЊ РїР°СЂР°Р»РµР»С–Р·РјСѓ')

    args = parser.parse_args()

    if not Path(args.image).exists():
        print(f"РџРѕРјРёР»РєР°: С„Р°Р№Р» {args.image} РЅРµ С–СЃРЅСѓС”")
        sys.exit(1)

    client = ModelClient(args.url)

    # РџРµСЂРµРІС–СЂРєР° Р·РґРѕСЂРѕРІ'СЏ СЃРµСЂРІРµСЂР°
    try:
        health_response = requests.get(f"{args.url}/health")
        if health_response.status_code != 200:
            print(f"РЎРµСЂРІРµСЂ РЅРµ РіРѕС‚РѕРІРёР№ РґРѕ СЂРѕР±РѕС‚Рё. РЎС‚Р°С‚СѓСЃ: {health_response.status_code}")
            sys.exit(1)
    except Exception as e:
        print(f"РќРµ РІРґР°Р»РѕСЃСЏ РїС–РґРєР»СЋС‡РёС‚РёСЃСЏ РґРѕ СЃРµСЂРІРµСЂР°: {e}")
        sys.exit(1)

    print(f"РЎРµСЂРІРµСЂ РіРѕС‚РѕРІРёР№ РґРѕ СЂРѕР±РѕС‚Рё. РџРѕС‡Р°С‚РѕРє С‚РµСЃС‚СѓРІР°РЅРЅСЏ...")
    run_concurrent_test(client, args.image, args.requests, args.concurrency)

