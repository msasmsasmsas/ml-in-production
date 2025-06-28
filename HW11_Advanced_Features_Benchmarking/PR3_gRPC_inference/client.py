# Updated version for PR
import os
import time
import argparse
import sys
from pathlib import Path
import grpc
import concurrent.futures

# Р†РјРїРѕСЂС‚СѓС”РјРѕ Р·РіРµРЅРµСЂРѕРІР°РЅС– gRPC РјРѕРґСѓР»С–
import inference_pb2
import inference_pb2_grpc

class InferenceClient:
    """
    РљР»С–С”РЅС‚ РґР»СЏ gRPC С–РЅС„РµСЂРµРЅСЃСѓ РјРѕРґРµР»РµР№ РјР°С€РёРЅРЅРѕРіРѕ РЅР°РІС‡Р°РЅРЅСЏ
    """
    def __init__(self, server_address):
        """
        Р†РЅС–С†С–Р°Р»С–Р·Р°С†С–СЏ РєР»С–С”РЅС‚Р°

        РџР°СЂР°РјРµС‚СЂРё:
        -----------
        server_address: Р°РґСЂРµСЃР° gRPC СЃРµСЂРІРµСЂР°
        """
        # РЎС‚РІРѕСЂРµРЅРЅСЏ РєР°РЅР°Р»Сѓ Р· РѕРїС†С–СЏРјРё РґР»СЏ РІРµР»РёРєРёС… РїРѕРІС–РґРѕРјР»РµРЅСЊ
        channel_options = [
            ('grpc.max_send_message_length', 50 * 1024 * 1024),  # 50 MB
            ('grpc.max_receive_message_length', 50 * 1024 * 1024),  # 50 MB
        ]
        self.channel = grpc.insecure_channel(server_address, options=channel_options)
        self.stub = inference_pb2_grpc.InferenceServiceStub(self.channel)

    def health_check(self):
        """
        РџРµСЂРµРІС–СЂРєР° СЃС‚Р°РЅСѓ СЃРµСЂРІРµСЂР°

        РџРѕРІРµСЂС‚Р°С”:
        -----------
        РѕР±'С”РєС‚ HealthCheckResponse
        """
        request = inference_pb2.HealthCheckRequest()
        return self.stub.HealthCheck(request)

    def predict(self, image_path):
        """
        Р’С–РґРїСЂР°РІР»СЏС” Р·Р°РїРёС‚ РЅР° РїСЂРѕРіРЅРѕР·СѓРІР°РЅРЅСЏ Р·РѕР±СЂР°Р¶РµРЅРЅСЏ

        РџР°СЂР°РјРµС‚СЂРё:
        -----------
        image_path: С€Р»СЏС… РґРѕ С„Р°Р№Р»Сѓ Р·РѕР±СЂР°Р¶РµРЅРЅСЏ

        РџРѕРІРµСЂС‚Р°С”:
        -----------
        РѕР±'С”РєС‚ PredictResponse С‚Р° С‡Р°СЃ РІРёРєРѕРЅР°РЅРЅСЏ Р·Р°РїРёС‚Сѓ
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

            return response, elapsed

        except Exception as e:
            print(f"РџРѕРјРёР»РєР° РїСЂРё РІРёРєРѕРЅР°РЅРЅС– Р·Р°РїРёС‚Сѓ: {e}")
            return None, 0

    def predict_stream(self, image_paths):
        """
        Р’С–РґРїСЂР°РІР»СЏС” РїРѕС‚РѕРєРѕРІРёР№ Р·Р°РїРёС‚ РЅР° РїСЂРѕРіРЅРѕР·СѓРІР°РЅРЅСЏ РєС–Р»СЊРєРѕС… Р·РѕР±СЂР°Р¶РµРЅСЊ

        РџР°СЂР°РјРµС‚СЂРё:
        -----------
        image_paths: СЃРїРёСЃРѕРє С€Р»СЏС…С–РІ РґРѕ С„Р°Р№Р»С–РІ Р·РѕР±СЂР°Р¶РµРЅСЊ

        РџРѕРІРµСЂС‚Р°С”:
        -----------
        РіРµРЅРµСЂР°С‚РѕСЂ РїР°СЂ (РІС–РґРїРѕРІС–РґСЊ, С‡Р°СЃ РІРёРєРѕРЅР°РЅРЅСЏ)
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
                    print(f"РџРѕРјРёР»РєР° РїСЂРё РїС–РґРіРѕС‚РѕРІС†С– Р·Р°РїРёС‚Сѓ РґР»СЏ {path}: {e}")

        try:
            start_time = time.time()
            responses = self.stub.PredictStream(request_generator())

            for response in responses:
                current_time = time.time()
                elapsed = current_time - start_time
                start_time = current_time

                yield response, elapsed

        except Exception as e:
            print(f"РџРѕРјРёР»РєР° РїСЂРё РІРёРєРѕРЅР°РЅРЅС– РїРѕС‚РѕРєРѕРІРѕРіРѕ Р·Р°РїРёС‚Сѓ: {e}")

    def close(self):
        """
        Р—Р°РєСЂРёС‚С‚СЏ Р·'С”РґРЅР°РЅРЅСЏ
        """
        self.channel.close()

def format_prediction(prediction):
    """
    Р¤РѕСЂРјР°С‚СѓРІР°РЅРЅСЏ РїСЂРѕРіРЅРѕР·Сѓ РґР»СЏ РІРёРІРµРґРµРЅРЅСЏ

    РџР°СЂР°РјРµС‚СЂРё:
    -----------
    prediction: РѕР±'С”РєС‚ ClassPrediction

    РџРѕРІРµСЂС‚Р°С”:
    -----------
    СЂСЏРґРѕРє Р· С„РѕСЂРјР°С‚РѕРІР°РЅРёРј РїСЂРѕРіРЅРѕР·РѕРј
    """
    return f"{prediction.class_name} ({prediction.class_id}): {prediction.score:.4f}"

def run_single_request(client, image_path):
    """
    Р’РёРєРѕРЅСѓС” РѕРґРёРЅ Р·Р°РїРёС‚ С– РІРёРІРѕРґРёС‚СЊ СЂРµР·СѓР»СЊС‚Р°С‚Рё

    РџР°СЂР°РјРµС‚СЂРё:
    -----------
    client: РµРєР·РµРјРїР»СЏСЂ InferenceClient
    image_path: С€Р»СЏС… РґРѕ С„Р°Р№Р»Сѓ Р·РѕР±СЂР°Р¶РµРЅРЅСЏ
    """
    print(f"Р’С–РґРїСЂР°РІР»РµРЅРЅСЏ Р·Р°РїРёС‚Сѓ РґР»СЏ {image_path}")

    response, elapsed = client.predict(image_path)

    if response is None:
        print("РќРµ РѕС‚СЂРёРјР°РЅРѕ РІС–РґРїРѕРІС–РґС– РІС–Рґ СЃРµСЂРІРµСЂР°")
        return

    print(f"\nР РµР·СѓР»СЊС‚Р°С‚Рё РїСЂРѕРіРЅРѕР·СѓРІР°РЅРЅСЏ:")
    print(f"ID Р·Р°РїРёС‚Сѓ: {response.request_id}")
    print(f"РЎС‚Р°С‚СѓСЃ: {'РЈСЃРїС–С€РЅРѕ' if response.success else 'РџРѕРјРёР»РєР°'}")

    if not response.success:
        print(f"РџРѕРјРёР»РєР°: {response.error}")
        return

    print(f"Р§Р°СЃ РѕР±СЂРѕР±РєРё РЅР° СЃРµСЂРІРµСЂС–: {response.processing_time:.2f} РјСЃ")
    print(f"Р—Р°РіР°Р»СЊРЅРёР№ С‡Р°СЃ Р·Р°РїРёС‚Сѓ: {elapsed*1000:.2f} РјСЃ")
    print(f"РњРµСЂРµР¶РµРІР° Р·Р°С‚СЂРёРјРєР°: {(elapsed*1000 - response.processing_time):.2f} РјСЃ")

    print("\nРўРѕРї-5 РїСЂРѕРіРЅРѕР·С–РІ:")
    for i, prediction in enumerate(response.predictions):
        print(f"{i+1}. {format_prediction(prediction)}")

    if response.metadata:
        print("\nРњРµС‚Р°РґР°РЅС–:")
        for key, value in response.metadata.items():
            print(f"{key}: {value}")

def run_benchmark(client, image_path, num_requests, concurrency):
    """
    Р’РёРєРѕРЅСѓС” benchmark Р· РІРёРєРѕСЂРёСЃС‚Р°РЅРЅСЏРј РїР°СЂР°Р»РµР»СЊРЅРёС… Р·Р°РїРёС‚С–РІ

    РџР°СЂР°РјРµС‚СЂРё:
    -----------
    client: РµРєР·РµРјРїР»СЏСЂ InferenceClient
    image_path: С€Р»СЏС… РґРѕ С„Р°Р№Р»Сѓ Р·РѕР±СЂР°Р¶РµРЅРЅСЏ
    num_requests: РєС–Р»СЊРєС–СЃС‚СЊ Р·Р°РїРёС‚С–РІ
    concurrency: РєС–Р»СЊРєС–СЃС‚СЊ РїР°СЂР°Р»РµР»СЊРЅРёС… Р·Р°РїРёС‚С–РІ
    """
    print(f"Р—Р°РїСѓСЃРє benchmark: {num_requests} Р·Р°РїРёС‚С–РІ Р· СЂС–РІРЅРµРј РїР°СЂР°Р»РµР»С–Р·РјСѓ {concurrency}")

    results = []
    errors = 0

    def send_request(image_path):
        response, elapsed = client.predict(image_path)
        if response is None or not response.success:
            return {'success': False, 'elapsed': elapsed, 'server_time': 0}
        return {
            'success': True, 
            'elapsed': elapsed, 
            'server_time': response.processing_time / 1000  # РєРѕРЅРІРµСЂС‚Р°С†С–СЏ Р· РјСЃ РІ СЃРµРєСѓРЅРґРё
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

    # РћР±С‡РёСЃР»РµРЅРЅСЏ СЃС‚Р°С‚РёСЃС‚РёРєРё
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

        print(f"\nР РµР·СѓР»СЊС‚Р°С‚Рё benchmark:")
        print(f"Р—Р°РіР°Р»СЊРЅРёР№ С‡Р°СЃ: {total_time:.2f} СЃ")
        print(f"РЈСЃРїС–С€РЅРёС… Р·Р°РїРёС‚С–РІ: {num_requests - errors} Р· {num_requests} ({100 * (num_requests - errors) / num_requests:.2f}%)")
        print(f"RPS (Р·Р°РїРёС‚С–РІ РЅР° СЃРµРєСѓРЅРґСѓ): {rps:.2f}")
        print(f"\nР§Р°СЃ РІРёРєРѕРЅР°РЅРЅСЏ Р·Р°РїРёС‚Сѓ (РєР»С–С”РЅС‚):")
        print(f"  РЎРµСЂРµРґРЅС–Р№: {avg_client_time * 1000:.2f} РјСЃ")
        print(f"  РњС–РЅС–РјР°Р»СЊРЅРёР№: {min_client_time * 1000:.2f} РјСЃ")
        print(f"  РњР°РєСЃРёРјР°Р»СЊРЅРёР№: {max_client_time * 1000:.2f} РјСЃ")
        print(f"  P95: {p95_client_time * 1000:.2f} РјСЃ")
        print(f"\nР§Р°СЃ РѕР±СЂРѕР±РєРё (СЃРµСЂРІРµСЂ):")
        print(f"  РЎРµСЂРµРґРЅС–Р№: {avg_server_time * 1000:.2f} РјСЃ")
        print(f"  РњС–РЅС–РјР°Р»СЊРЅРёР№: {min_server_time * 1000:.2f} РјСЃ")
        print(f"  РњР°РєСЃРёРјР°Р»СЊРЅРёР№: {max_server_time * 1000:.2f} РјСЃ")
        print(f"  P95: {p95_server_time * 1000:.2f} РјСЃ")
        print(f"\nРњРµСЂРµР¶РµРІР° Р·Р°С‚СЂРёРјРєР° (С‚СѓРґРё-РЅР°Р·Р°Рґ):")
        print(f"  РЎРµСЂРµРґРЅСЏ: {(avg_client_time - avg_server_time) * 1000:.2f} РјСЃ")

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
        print("РќРµ РѕС‚СЂРёРјР°РЅРѕ СѓСЃРїС–С€РЅРёС… СЂРµР·СѓР»СЊС‚Р°С‚С–РІ")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='gRPC РєР»С–С”РЅС‚ РґР»СЏ С–РЅС„РµСЂРµРЅСЃСѓ РјРѕРґРµР»РµР№')
    parser.add_argument('--server', type=str, default='localhost:50051', help='РђРґСЂРµСЃР° gRPC СЃРµСЂРІРµСЂР°')
    parser.add_argument('--image', type=str, default='test_image.jpg', help='РЁР»СЏС… РґРѕ С‚РµСЃС‚РѕРІРѕРіРѕ Р·РѕР±СЂР°Р¶РµРЅРЅСЏ')
    parser.add_argument('--mode', type=str, choices=['single', 'benchmark'], default='single', 
                        help='Р РµР¶РёРј СЂРѕР±РѕС‚Рё: РѕРґРёРЅРѕС‡РЅРёР№ Р·Р°РїРёС‚ (single) Р°Р±Рѕ С‚РµСЃС‚СѓРІР°РЅРЅСЏ РїСЂРѕРґСѓРєС‚РёРІРЅРѕСЃС‚С– (benchmark)')
    parser.add_argument('--requests', type=int, default=100, help='РљС–Р»СЊРєС–СЃС‚СЊ Р·Р°РїРёС‚С–РІ РґР»СЏ СЂРµР¶РёРјСѓ benchmark')
    parser.add_argument('--concurrency', type=int, default=10, help='Р С–РІРµРЅСЊ РїР°СЂР°Р»РµР»С–Р·РјСѓ РґР»СЏ СЂРµР¶РёРјСѓ benchmark')

    args = parser.parse_args()

    if not Path(args.image).exists():
        print(f"РџРѕРјРёР»РєР°: С„Р°Р№Р» {args.image} РЅРµ С–СЃРЅСѓС”")
        sys.exit(1)

    client = InferenceClient(args.server)

    try:
        # РџРµСЂРµРІС–СЂРєР° Р·РґРѕСЂРѕРІ'СЏ СЃРµСЂРІРµСЂР°
        health_response = client.health_check()
        print(f"РЎС‚Р°С‚СѓСЃ СЃРµСЂРІРµСЂР°: {health_response.status}")

        if health_response.status != inference_pb2.ServingStatus.SERVING:
            print("РЎРµСЂРІРµСЂ РЅРµ РіРѕС‚РѕРІРёР№ РґРѕ СЂРѕР±РѕС‚Рё")
            sys.exit(1)

        print("РЎРµСЂРІРµСЂ РіРѕС‚РѕРІРёР№ РґРѕ СЂРѕР±РѕС‚Рё")
        if health_response.metadata:
            print("Р†РЅС„РѕСЂРјР°С†С–СЏ РїСЂРѕ СЃРµСЂРІРµСЂ:")
            for key, value in health_response.metadata.items():
                print(f"{key}: {value}")

        # Р’РёРєРѕРЅР°РЅРЅСЏ Р·Р°РїРёС‚С–РІ РІС–РґРїРѕРІС–РґРЅРѕ РґРѕ СЂРµР¶РёРјСѓ
        if args.mode == 'single':
            run_single_request(client, args.image)
        else:  # benchmark
            run_benchmark(client, args.image, args.requests, args.concurrency)

    except Exception as e:
        print(f"РџРѕРјРёР»РєР°: {e}")
        sys.exit(1)
    finally:
        client.close()

