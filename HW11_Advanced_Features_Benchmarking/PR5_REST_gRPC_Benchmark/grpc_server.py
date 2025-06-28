# Updated version for PR
import os
import io
import json
import time
import uuid
import grpc
import torch
import argparse
import numpy as np
import concurrent.futures
from PIL import Image
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights

# Р†РјРїРѕСЂС‚СѓС”РјРѕ Р·РіРµРЅРµСЂРѕРІР°РЅС– gRPC РјРѕРґСѓР»С–
import inference_pb2
import inference_pb2_grpc

class InferenceServicer(inference_pb2_grpc.InferenceServiceServicer):
    """
    РЎРµСЂРІС–СЃ РґР»СЏ gRPC С–РЅС„РµСЂРµРЅСЃСѓ РјРѕРґРµР»РµР№ РјР°С€РёРЅРЅРѕРіРѕ РЅР°РІС‡Р°РЅРЅСЏ
    """
    def __init__(self, device=None):
        """
        Р†РЅС–С†С–Р°Р»С–Р·Р°С†С–СЏ СЃРµСЂРІС–СЃСѓ

        РџР°СЂР°РјРµС‚СЂРё:
        -----------
        device: РїСЂРёСЃС‚СЂС–Р№ РґР»СЏ РІРёРєРѕРЅР°РЅРЅСЏ (cuda Р°Р±Рѕ cpu)
        """
        # Р’РёР·РЅР°С‡РµРЅРЅСЏ РїСЂРёСЃС‚СЂРѕСЋ
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        print(f"Р†РЅС–С†С–Р°Р»С–Р·Р°С†С–СЏ СЃРµСЂРІС–СЃСѓ РЅР° РїСЂРёСЃС‚СЂРѕС—: {self.device}")

        # Р—Р°РІР°РЅС‚Р°Р¶РµРЅРЅСЏ РјРѕРґРµР»С–
        self.model = self._load_model()

        # РЎС‚РІРѕСЂРµРЅРЅСЏ С‚СЂР°РЅСЃС„РѕСЂРјР°С†С–Р№ РґР»СЏ Р·РѕР±СЂР°Р¶РµРЅСЊ
        self.preprocessing = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Р—Р°РІР°РЅС‚Р°Р¶РµРЅРЅСЏ РєР»Р°СЃС–РІ ImageNet
        with open('imagenet_classes.json', 'r') as f:
            self.labels = json.load(f)

    def _load_model(self):
        """
        Р—Р°РІР°РЅС‚Р°Р¶РµРЅРЅСЏ РїРѕРїРµСЂРµРґРЅСЊРѕ РЅР°РІС‡РµРЅРѕС— РјРѕРґРµР»С– ResNet50

        РџРѕРІРµСЂС‚Р°С”:
        -----------
        Р·Р°РІР°РЅС‚Р°Р¶РµРЅР° РјРѕРґРµР»СЊ
        """
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        model.to(self.device)
        model.eval()
        return model

    def _preprocess_image(self, image_bytes):
        """
        РџРѕРїРµСЂРµРґРЅСЏ РѕР±СЂРѕР±РєР° Р·РѕР±СЂР°Р¶РµРЅРЅСЏ

        РџР°СЂР°РјРµС‚СЂРё:
        -----------
        image_bytes: Р±Р°Р№С‚Рё Р·РѕР±СЂР°Р¶РµРЅРЅСЏ

        РџРѕРІРµСЂС‚Р°С”:
        -----------
        С‚РµРЅР·РѕСЂ Р·РѕР±СЂР°Р¶РµРЅРЅСЏ
        """
        img = Image.open(io.BytesIO(image_bytes))
        img_tensor = self.preprocessing(img).unsqueeze(0)  # Р”РѕРґР°С”РјРѕ РІРёРјС–СЂРЅС–СЃС‚СЊ Р±Р°С‚С‡Сѓ
        img_tensor = img_tensor.to(self.device)
        return img_tensor

    def _process_prediction(self, outputs, top_k=5):
        """
        РћР±СЂРѕР±РєР° СЂРµР·СѓР»СЊС‚Р°С‚С–РІ РїСЂРѕРіРЅРѕР·СѓРІР°РЅРЅСЏ

        РџР°СЂР°РјРµС‚СЂРё:
        -----------
        outputs: РІРёРІС–Рґ РјРѕРґРµР»С–
        top_k: РєС–Р»СЊРєС–СЃС‚СЊ РЅР°Р№РєСЂР°С‰РёС… РєР»Р°СЃС–РІ РґР»СЏ РїРѕРІРµСЂРЅРµРЅРЅСЏ

        РџРѕРІРµСЂС‚Р°С”:
        -----------
        СЃРїРёСЃРѕРє РѕР±'С”РєС‚С–РІ ClassPrediction
        """
        # Р—Р°СЃС‚РѕСЃСѓРІР°РЅРЅСЏ softmax РґР»СЏ РѕС‚СЂРёРјР°РЅРЅСЏ Р№РјРѕРІС–СЂРЅРѕСЃС‚РµР№
        probs = torch.nn.functional.softmax(outputs, dim=1)[0]

        # РћС‚СЂРёРјР°РЅРЅСЏ top_k РЅР°Р№РєСЂР°С‰РёС… СЂРµР·СѓР»СЊС‚Р°С‚С–РІ
        top_probs, top_indices = torch.topk(probs, top_k)

        # РџРµСЂРµС‚РІРѕСЂРµРЅРЅСЏ РІ numpy РґР»СЏ Р»РµРіС€РѕС— РѕР±СЂРѕР±РєРё
        top_probs = top_probs.cpu().numpy()
        top_indices = top_indices.cpu().numpy()

        # РЎС‚РІРѕСЂРµРЅРЅСЏ СЃРїРёСЃРєСѓ РїСЂРѕРіРЅРѕР·С–РІ
        predictions = []
        for i, (idx, prob) in enumerate(zip(top_indices, top_probs)):
            prediction = inference_pb2.ClassPrediction(
                class_id=int(idx),
                class_name=self.labels.get(str(idx), f"Unknown class {idx}"),
                score=float(prob)
            )
            predictions.append(prediction)

        return predictions

    def Predict(self, request, context):
        """
        Р РµР°Р»С–Р·Р°С†С–СЏ RPC РјРµС‚РѕРґСѓ Predict

        РџР°СЂР°РјРµС‚СЂРё:
        -----------
        request: РѕР±'С”РєС‚ PredictRequest
        context: РєРѕРЅС‚РµРєСЃС‚ gRPC

        РџРѕРІРµСЂС‚Р°С”:
        -----------
        РѕР±'С”РєС‚ PredictResponse
        """
        request_id = str(uuid.uuid4())
        start_time = time.time()

        try:
            # РџРµСЂРµРІС–СЂРєР° РЅР°СЏРІРЅРѕСЃС‚С– РґР°РЅРёС…
            if not request.data:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details("Р’С–РґСЃСѓС‚РЅС– РґР°РЅС– Р·РѕР±СЂР°Р¶РµРЅРЅСЏ")
                return inference_pb2.PredictResponse(
                    request_id=request_id,
                    success=False,
                    error="Р’С–РґСЃСѓС‚РЅС– РґР°РЅС– Р·РѕР±СЂР°Р¶РµРЅРЅСЏ"
                )

            # РџРѕРїРµСЂРµРґРЅСЏ РѕР±СЂРѕР±РєР° Р·РѕР±СЂР°Р¶РµРЅРЅСЏ
            img_tensor = self._preprocess_image(request.data)

            # РџСЂРѕРіРЅРѕР·СѓРІР°РЅРЅСЏ
            with torch.no_grad():
                outputs = self.model(img_tensor)

            # РћР±СЂРѕР±РєР° СЂРµР·СѓР»СЊС‚Р°С‚С–РІ
            predictions = self._process_prediction(outputs)

            # РћР±С‡РёСЃР»РµРЅРЅСЏ С‡Р°СЃСѓ РѕР±СЂРѕР±РєРё
            processing_time = (time.time() - start_time) * 1000  # РјСЃ

            # РџС–РґРіРѕС‚РѕРІРєР° РІС–РґРїРѕРІС–РґС–
            response = inference_pb2.PredictResponse(
                request_id=request_id,
                success=True,
                predictions=predictions,
                processing_time=processing_time,
                metadata={
                    'model': 'ResNet50',
                    'framework': 'PyTorch',
                    'device': self.device.type
                }
            )

            return response

        except Exception as e:
            # РћР±СЂРѕР±РєР° РїРѕРјРёР»РѕРє
            processing_time = (time.time() - start_time) * 1000  # РјСЃ

            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))

            return inference_pb2.PredictResponse(
                request_id=request_id,
                success=False,
                error=str(e),
                processing_time=processing_time
            )

    def PredictStream(self, request_iterator, context):
        """
        Р РµР°Р»С–Р·Р°С†С–СЏ РїРѕС‚РѕРєРѕРІРѕРіРѕ RPC РјРµС‚РѕРґСѓ PredictStream

        РџР°СЂР°РјРµС‚СЂРё:
        -----------
        request_iterator: С–С‚РµСЂР°С‚РѕСЂ Р·Р°РїРёС‚С–РІ PredictRequest
        context: РєРѕРЅС‚РµРєСЃС‚ gRPC

        РџРѕРІРµСЂС‚Р°С”:
        -----------
        С–С‚РµСЂР°С‚РѕСЂ РІС–РґРїРѕРІС–РґРµР№ PredictResponse
        """
        for request in request_iterator:
            yield self.Predict(request, context)

    def HealthCheck(self, request, context):
        """
        Р РµР°Р»С–Р·Р°С†С–СЏ RPC РјРµС‚РѕРґСѓ HealthCheck

        РџР°СЂР°РјРµС‚СЂРё:
        -----------
        request: РѕР±'С”РєС‚ HealthCheckRequest
        context: РєРѕРЅС‚РµРєСЃС‚ gRPC

        РџРѕРІРµСЂС‚Р°С”:
        -----------
        РѕР±'С”РєС‚ HealthCheckResponse
        """
        return inference_pb2.HealthCheckResponse(
            status=inference_pb2.ServingStatus.SERVING,
            metadata={
                'model': 'ResNet50',
                'framework': 'PyTorch',
                'version': '1.0.0',
                'device': self.device.type
            }
        )

def serve(host='[::]', port=50051, max_workers=10, device=None):
    """
    Р—Р°РїСѓСЃРє gRPC СЃРµСЂРІРµСЂР°

    РџР°СЂР°РјРµС‚СЂРё:
    -----------
    host: С…РѕСЃС‚ РґР»СЏ РїСЂРѕСЃР»СѓС…РѕРІСѓРІР°РЅРЅСЏ
    port: РїРѕСЂС‚ РґР»СЏ РїСЂРѕСЃР»СѓС…РѕРІСѓРІР°РЅРЅСЏ
    max_workers: РјР°РєСЃРёРјР°Р»СЊРЅР° РєС–Р»СЊРєС–СЃС‚СЊ РїРѕС‚РѕРєС–РІ РґР»СЏ РѕР±СЂРѕР±РєРё Р·Р°РїРёС‚С–РІ
    device: РїСЂРёСЃС‚СЂС–Р№ РґР»СЏ РІРёРєРѕРЅР°РЅРЅСЏ (cuda Р°Р±Рѕ cpu)
    """
    server = grpc.server(
        concurrent.futures.ThreadPoolExecutor(max_workers=max_workers),
        options=[
            ('grpc.max_send_message_length', 50 * 1024 * 1024),  # 50 MB
            ('grpc.max_receive_message_length', 50 * 1024 * 1024),  # 50 MB
        ]
    )

    inference_pb2_grpc.add_InferenceServiceServicer_to_server(
        InferenceServicer(device=device), server
    )

    server_address = f'{host}:{port}'
    server.add_insecure_port(server_address)
    server.start()

    print(f"gRPC СЃРµСЂРІРµСЂ Р·Р°РїСѓС‰РµРЅРѕ РЅР° {server_address}")

    try:
        # РЎРµСЂРІРµСЂ РїСЂР°С†СЋС” РґРѕ РїРµСЂРµСЂРёРІР°РЅРЅСЏ (Ctrl+C)
        server.wait_for_termination()
    except KeyboardInterrupt:
        server.stop(0)
        print("РЎРµСЂРІРµСЂ Р·СѓРїРёРЅРµРЅРѕ")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='gRPC СЃРµСЂРІРµСЂ РґР»СЏ С–РЅС„РµСЂРµРЅСЃСѓ РјРѕРґРµР»РµР№')
    parser.add_argument('--host', type=str, default='[::]', help='РҐРѕСЃС‚ РґР»СЏ РїСЂРѕСЃР»СѓС…РѕРІСѓРІР°РЅРЅСЏ')
    parser.add_argument('--port', type=int, default=50051, help='РџРѕСЂС‚ РґР»СЏ РїСЂРѕСЃР»СѓС…РѕРІСѓРІР°РЅРЅСЏ')
    parser.add_argument('--workers', type=int, default=10, help='РљС–Р»СЊРєС–СЃС‚СЊ РїРѕС‚РѕРєС–РІ РґР»СЏ РѕР±СЂРѕР±РєРё Р·Р°РїРёС‚С–РІ')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], default=None,
                        help='РџСЂРёСЃС‚СЂС–Р№ РґР»СЏ РІРёРєРѕРЅР°РЅРЅСЏ (Р·Р° Р·Р°РјРѕРІС‡СѓРІР°РЅРЅСЏРј: Р°РІС‚РѕРІРёР·РЅР°С‡РµРЅРЅСЏ)')

    args = parser.parse_args()

    serve(host=args.host, port=args.port, max_workers=args.workers, device=args.device)

