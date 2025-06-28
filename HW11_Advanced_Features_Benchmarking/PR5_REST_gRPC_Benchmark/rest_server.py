import os
import io
import json
import time
import uuid
import torch
import argparse
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
from flask import Flask, request, jsonify
from waitress import serve

app = Flask(__name__)

# Глобальні змінні для моделі та трансформацій
model = None
preprocessing = None
labels = None

def load_model(device=None):
    """
    Завантаження попередньо навченої моделі ResNet50

    Параметри:
    -----------
    device: пристрій для виконання (cuda або cpu)

    Повертає:
    -----------
    завантажена модель
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)

    print(f"Завантаження моделі на пристрій: {device}")

    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    model.to(device)
    model.eval()

    return model

def setup():
    """
    Ініціалізація моделі, трансформацій та класів
    """
    global model, preprocessing, labels

    # Завантаження моделі
    model = load_model()

    # Створення трансформацій для зображень
    preprocessing = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Завантаження класів ImageNet
    with open('imagenet_classes.json', 'r') as f:
        labels = json.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    """
    Ендпоінт для прогнозування зображення

    Повертає:
    -----------
    JSON з результатами прогнозування
    """
    request_id = str(uuid.uuid4())
    start_time = time.time()

    try:
        # Перевірка наявності файлу в запиті
        if 'file' not in request.files:
            return jsonify({
                'request_id': request_id,
                'success': False,
                'error': 'Файл не знайдено в запиті',
                'processing_time': (time.time() - start_time) * 1000
            }), 400

        # Отримання файлу
        file = request.files['file']
        img_bytes = file.read()

        # Завантаження та попередня обробка зображення
        img = Image.open(io.BytesIO(img_bytes))
        img_tensor = preprocessing(img).unsqueeze(0)  # додавання вимірності батчу

        # Переміщення тензора на GPU, якщо доступно
        device = next(model.parameters()).device
        img_tensor = img_tensor.to(device)

        # Прогнозування
        with torch.no_grad():
            outputs = model(img_tensor)

        # Обробка результатів
        probs = torch.nn.functional.softmax(outputs, dim=1)[0]
        top5_probs, top5_indices = torch.topk(probs, 5)

        # Підготовка відповіді
        predictions = []
        for i, (idx, prob) in enumerate(zip(top5_indices.cpu().numpy(), top5_probs.cpu().numpy())):
            predictions.append({
                'class_id': int(idx),
                'class_name': labels.get(str(idx), f"Unknown class {idx}"),
                'score': float(prob)
            })

        # Час обробки в мілісекундах
        processing_time = (time.time() - start_time) * 1000

        return jsonify({
            'request_id': request_id,
            'success': True,
            'predictions': predictions,
            'processing_time': processing_time,
            'metadata': {
                'model': 'ResNet50',
                'framework': 'PyTorch',
                'device': device.type
            }
        })

    except Exception as e:
        # Обробка помилок
        processing_time = (time.time() - start_time) * 1000

        return jsonify({
            'request_id': request_id,
            'success': False,
            'error': str(e),
            'processing_time': processing_time
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """
    Ендпоінт для перевірки стану сервера

    Повертає:
    -----------
    JSON зі статусом сервера
    """
    # Перевірка стану моделі
    if model is None:
        return jsonify({
            'status': 'not_ready',
            'error': 'Модель не завантажена'
        }), 503

    # Отримання інформації про пристрій
    device = next(model.parameters()).device

    return jsonify({
        'status': 'ok',
        'model': 'ResNet50',
        'framework': 'PyTorch',
        'device': device.type,
        'version': '1.0.0'
    })

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='REST сервер для інференсу моделей')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Хост для прослуховування')
    parser.add_argument('--port', type=int, default=5000, help='Порт для прослуховування')
    parser.add_argument('--debug', action='store_true', help='Режим відлагодження Flask')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], default=None, 
                        help='Пристрій для виконання (за замовчуванням: автовизначення)')

    args = parser.parse_args()

    # Ініціалізація моделі та інших компонентів
    setup()

    if args.debug:
        # Запуск у режимі відлагодження (не рекомендується для продакшену)
        app.run(host=args.host, port=args.port, debug=True)
    else:
        # Запуск через Waitress для кращої продуктивності
        print(f"Запуск сервера на {args.host}:{args.port}")
        serve(app, host=args.host, port=args.port, threads=8)
