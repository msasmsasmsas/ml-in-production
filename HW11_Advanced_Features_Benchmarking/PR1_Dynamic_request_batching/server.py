import os
import io
import json
import torch
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, request, jsonify
from torchvision.models import resnet50, ResNet50_Weights
from model_batcher import DynamicBatcher

app = Flask(__name__)

# Ініціалізація моделі
def load_model():
    """
    Завантаження попередньо навченої моделі ResNet50
    """
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    return model

# Створення моделі та батчера
model = load_model()
batcher = DynamicBatcher(model, max_batch_size=16, max_wait_time=0.1)

# Підготовка трансформацій для зображень
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
    Ендпоінт для прогнозування з використанням динамічного батчингу
    """
    if 'file' not in request.files:
        return jsonify({'error': 'Немає файлу в запиті'}), 400

    file = request.files['file']
    img_bytes = file.read()

    try:
        # Попередня обробка зображення
        img = Image.open(io.BytesIO(img_bytes))
        img_tensor = preprocessing(img)

        # Використання динамічного батчингу для прогнозування
        outputs = batcher.predict(img_tensor)

        # Обробка результатів
        _, indices = torch.topk(torch.from_numpy(outputs), 5)
        top_predictions = [
            {'class_id': int(idx), 'class_name': labels[str(idx)], 'score': float(outputs[idx])}
            for idx in indices
        ]

        return jsonify({
            'success': True,
            'predictions': top_predictions
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """
    Ендпоінт для перевірки стану сервера
    """
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
