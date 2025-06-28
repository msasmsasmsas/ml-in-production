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

# Імпортуємо згенеровані gRPC модулі
import inference_pb2
import inference_pb2_grpc

class InferenceServicer(inference_pb2_grpc.InferenceServiceServicer):
    """
    Сервіс для gRPC інференсу моделей машинного навчання
    """
    def __init__(self, device=None):
        """
        Ініціалізація сервісу

        Параметри:
        -----------
        device: пристрій для виконання (cuda або cpu)
        """
        # Визначення пристрою
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        print(f"Ініціалізація сервісу на пристрої: {self.device}")

        # Завантаження моделі
        self.model = self._load_model()

        # Створення трансформацій для зображень
        self.preprocessing = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Завантаження класів ImageNet
        with open('imagenet_classes.json', 'r') as f:
            self.labels = json.load(f)

    def _load_model(self):
        """
        Завантаження попередньо навченої моделі ResNet50

        Повертає:
        -----------
        завантажена модель
        """
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        model.to(self.device)
        model.eval()
        return model

    def _preprocess_image(self, image_bytes):
        """
        Попередня обробка зображення

        Параметри:
        -----------
        image_bytes: байти зображення

        Повертає:
        -----------
        тензор зображення
        """
        img = Image.open(io.BytesIO(image_bytes))
        img_tensor = self.preprocessing(img).unsqueeze(0)  # Додаємо вимірність батчу
        img_tensor = img_tensor.to(self.device)
        return img_tensor

    def _process_prediction(self, outputs, top_k=5):
        """
        Обробка результатів прогнозування

        Параметри:
        -----------
        outputs: вивід моделі
        top_k: кількість найкращих класів для повернення

        Повертає:
        -----------
        список об'єктів ClassPrediction
        """
        # Застосування softmax для отримання ймовірностей
        probs = torch.nn.functional.softmax(outputs, dim=1)[0]

        # Отримання top_k найкращих результатів
        top_probs, top_indices = torch.topk(probs, top_k)

        # Перетворення в numpy для легшої обробки
        top_probs = top_probs.cpu().numpy()
        top_indices = top_indices.cpu().numpy()

        # Створення списку прогнозів
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
        Реалізація RPC методу Predict

        Параметри:
        -----------
        request: об'єкт PredictRequest
        context: контекст gRPC

        Повертає:
        -----------
        об'єкт PredictResponse
        """
        request_id = str(uuid.uuid4())
        start_time = time.time()

        try:
            # Перевірка наявності даних
            if not request.data:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details("Відсутні дані зображення")
                return inference_pb2.PredictResponse(
                    request_id=request_id,
                    success=False,
                    error="Відсутні дані зображення"
                )

            # Попередня обробка зображення
            img_tensor = self._preprocess_image(request.data)

            # Прогнозування
            with torch.no_grad():
                outputs = self.model(img_tensor)

            # Обробка результатів
            predictions = self._process_prediction(outputs)

            # Обчислення часу обробки
            processing_time = (time.time() - start_time) * 1000  # мс

            # Підготовка відповіді
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
            # Обробка помилок
            processing_time = (time.time() - start_time) * 1000  # мс

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
        Реалізація потокового RPC методу PredictStream

        Параметри:
        -----------
        request_iterator: ітератор запитів PredictRequest
        context: контекст gRPC

        Повертає:
        -----------
        ітератор відповідей PredictResponse
        """
        for request in request_iterator:
            yield self.Predict(request, context)

    def HealthCheck(self, request, context):
        """
        Реалізація RPC методу HealthCheck

        Параметри:
        -----------
        request: об'єкт HealthCheckRequest
        context: контекст gRPC

        Повертає:
        -----------
        об'єкт HealthCheckResponse
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
    Запуск gRPC сервера

    Параметри:
    -----------
    host: хост для прослуховування
    port: порт для прослуховування
    max_workers: максимальна кількість потоків для обробки запитів
    device: пристрій для виконання (cuda або cpu)
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

    print(f"gRPC сервер запущено на {server_address}")

    try:
        # Сервер працює до переривання (Ctrl+C)
        server.wait_for_termination()
    except KeyboardInterrupt:
        server.stop(0)
        print("Сервер зупинено")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='gRPC сервер для інференсу моделей')
    parser.add_argument('--host', type=str, default='[::]', help='Хост для прослуховування')
    parser.add_argument('--port', type=int, default=50051, help='Порт для прослуховування')
    parser.add_argument('--workers', type=int, default=10, help='Кількість потоків для обробки запитів')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], default=None,
                        help='Пристрій для виконання (за замовчуванням: автовизначення)')

    args = parser.parse_args()

    serve(host=args.host, port=args.port, max_workers=args.workers, device=args.device)
