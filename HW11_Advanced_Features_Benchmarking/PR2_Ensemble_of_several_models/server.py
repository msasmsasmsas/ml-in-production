import os
import io
import json
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
from flask import Flask, request, jsonify
from model_ensemble import ModelEnsemble

app = Flask(__name__)

# Ініціалізація моделей для ансамблю
def load_models():
    """
    Завантаження попередньо навчених моделей для ансамблю
    """
    model1 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model2 = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
    model3 = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)

    # Переведення моделей у режим оцінки
    model1.eval()
    model2.eval()
    model3.eval()

    # Переміщення моделей на GPU, якщо доступно
    if torch.cuda.is_available():
        model1 = model1.cuda()
        model2 = model2.cuda()
        model3 = model3.cuda()

    return [model1, model2, model3]

# Створення ансамблю моделей
models_list = load_models()
ensemble = ModelEnsemble(
    models=models_list,
    weights=[0.4, 0.3, 0.3],  # Вагові коефіцієнти для кожної моделі
    aggregation_method='softmax_average'  # Метод агрегації
)

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
    Ендпоінт для прогнозування з використанням ансамблю моделей
    """
    if 'file' not in request.files:
        return jsonify({'error': 'Немає файлу в запиті'}), 400

    file = request.files['file']
    img_bytes = file.read()

    try:
        # Попередня обробка зображення
        img = Image.open(io.BytesIO(img_bytes))
        img_tensor = preprocessing(img).unsqueeze(0)  # додавання вимірності батчу

        if torch.cuda.is_available():
            img_tensor = img_tensor.cuda()

        # Прогнозування ансамблем моделей
        ensemble_outputs = ensemble(img_tensor)

        # Отримання прогнозів від окремих моделей, якщо запитано
        individual_predictions = None
        if request.args.get('include_individual', '').lower() == 'true':
            individual_outputs = ensemble.get_individual_predictions(img_tensor)
            individual_predictions = []

            for i, output in enumerate(individual_outputs):
                # Отримання топ-5 класів для кожної моделі
                probs = torch.nn.functional.softmax(output, dim=1)[0]
                top5_probs, top5_indices = torch.topk(probs, 5)

                model_predictions = [
                    {'class_id': int(idx), 'class_name': labels[str(idx)], 'score': float(prob)}
                    for idx, prob in zip(top5_indices.cpu().numpy(), top5_probs.cpu().numpy())
                ]

                individual_predictions.append({
                    'model_id': i,
                    'model_name': type(ensemble.models[i]).__name__,
                    'weight': ensemble.weights[i],
                    'predictions': model_predictions
                })

        # Обробка результатів ансамблю
        probs = torch.nn.functional.softmax(ensemble_outputs, dim=1)[0]
        top5_probs, top5_indices = torch.topk(probs, 5)

        ensemble_predictions = [
            {'class_id': int(idx), 'class_name': labels[str(idx)], 'score': float(prob)}
            for idx, prob in zip(top5_indices.cpu().numpy(), top5_probs.cpu().numpy())
        ]

        response = {
            'success': True,
            'ensemble_predictions': ensemble_predictions,
            'aggregation_method': ensemble.aggregation_method
        }

        if individual_predictions:
            response['individual_predictions'] = individual_predictions

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """
    Ендпоінт для перевірки стану сервера
    """
    return jsonify({
        'status': 'ok',
        'models_loaded': len(ensemble.models),
        'aggregation_method': ensemble.aggregation_method
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
