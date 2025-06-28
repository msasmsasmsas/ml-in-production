# Updated version for PR
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

# Р†РЅС–С†С–Р°Р»С–Р·Р°С†С–СЏ РјРѕРґРµР»РµР№ РґР»СЏ Р°РЅСЃР°РјР±Р»СЋ
def load_models():
    """
    Р—Р°РІР°РЅС‚Р°Р¶РµРЅРЅСЏ РїРѕРїРµСЂРµРґРЅСЊРѕ РЅР°РІС‡РµРЅРёС… РјРѕРґРµР»РµР№ РґР»СЏ Р°РЅСЃР°РјР±Р»СЋ
    """
    model1 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model2 = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
    model3 = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)

    # РџРµСЂРµРІРµРґРµРЅРЅСЏ РјРѕРґРµР»РµР№ Сѓ СЂРµР¶РёРј РѕС†С–РЅРєРё
    model1.eval()
    model2.eval()
    model3.eval()

    # РџРµСЂРµРјС–С‰РµРЅРЅСЏ РјРѕРґРµР»РµР№ РЅР° GPU, СЏРєС‰Рѕ РґРѕСЃС‚СѓРїРЅРѕ
    if torch.cuda.is_available():
        model1 = model1.cuda()
        model2 = model2.cuda()
        model3 = model3.cuda()

    return [model1, model2, model3]

# РЎС‚РІРѕСЂРµРЅРЅСЏ Р°РЅСЃР°РјР±Р»СЋ РјРѕРґРµР»РµР№
models_list = load_models()
ensemble = ModelEnsemble(
    models=models_list,
    weights=[0.4, 0.3, 0.3],  # Р’Р°РіРѕРІС– РєРѕРµС„С–С†С–С”РЅС‚Рё РґР»СЏ РєРѕР¶РЅРѕС— РјРѕРґРµР»С–
    aggregation_method='softmax_average'  # РњРµС‚РѕРґ Р°РіСЂРµРіР°С†С–С—
)

# РџС–РґРіРѕС‚РѕРІРєР° С‚СЂР°РЅСЃС„РѕСЂРјР°С†С–Р№ РґР»СЏ Р·РѕР±СЂР°Р¶РµРЅСЊ
preprocessing = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Р—Р°РІР°РЅС‚Р°Р¶РµРЅРЅСЏ РєР»Р°СЃС–РІ ImageNet
with open('imagenet_classes.json', 'r') as f:
    labels = json.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    """
    Р•РЅРґРїРѕС–РЅС‚ РґР»СЏ РїСЂРѕРіРЅРѕР·СѓРІР°РЅРЅСЏ Р· РІРёРєРѕСЂРёСЃС‚Р°РЅРЅСЏРј Р°РЅСЃР°РјР±Р»СЋ РјРѕРґРµР»РµР№
    """
    if 'file' not in request.files:
        return jsonify({'error': 'РќРµРјР°С” С„Р°Р№Р»Сѓ РІ Р·Р°РїРёС‚С–'}), 400

    file = request.files['file']
    img_bytes = file.read()

    try:
        # РџРѕРїРµСЂРµРґРЅСЏ РѕР±СЂРѕР±РєР° Р·РѕР±СЂР°Р¶РµРЅРЅСЏ
        img = Image.open(io.BytesIO(img_bytes))
        img_tensor = preprocessing(img).unsqueeze(0)  # РґРѕРґР°РІР°РЅРЅСЏ РІРёРјС–СЂРЅРѕСЃС‚С– Р±Р°С‚С‡Сѓ

        if torch.cuda.is_available():
            img_tensor = img_tensor.cuda()

        # РџСЂРѕРіРЅРѕР·СѓРІР°РЅРЅСЏ Р°РЅСЃР°РјР±Р»РµРј РјРѕРґРµР»РµР№
        ensemble_outputs = ensemble(img_tensor)

        # РћС‚СЂРёРјР°РЅРЅСЏ РїСЂРѕРіРЅРѕР·С–РІ РІС–Рґ РѕРєСЂРµРјРёС… РјРѕРґРµР»РµР№, СЏРєС‰Рѕ Р·Р°РїРёС‚Р°РЅРѕ
        individual_predictions = None
        if request.args.get('include_individual', '').lower() == 'true':
            individual_outputs = ensemble.get_individual_predictions(img_tensor)
            individual_predictions = []

            for i, output in enumerate(individual_outputs):
                # РћС‚СЂРёРјР°РЅРЅСЏ С‚РѕРї-5 РєР»Р°СЃС–РІ РґР»СЏ РєРѕР¶РЅРѕС— РјРѕРґРµР»С–
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

        # РћР±СЂРѕР±РєР° СЂРµР·СѓР»СЊС‚Р°С‚С–РІ Р°РЅСЃР°РјР±Р»СЋ
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
    Р•РЅРґРїРѕС–РЅС‚ РґР»СЏ РїРµСЂРµРІС–СЂРєРё СЃС‚Р°РЅСѓ СЃРµСЂРІРµСЂР°
    """
    return jsonify({
        'status': 'ok',
        'models_loaded': len(ensemble.models),
        'aggregation_method': ensemble.aggregation_method
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)

