# Updated version for PR
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

# Р†РЅС–С†С–Р°Р»С–Р·Р°С†С–СЏ РјРѕРґРµР»С–
def load_model():
    """
    Р—Р°РІР°РЅС‚Р°Р¶РµРЅРЅСЏ РїРѕРїРµСЂРµРґРЅСЊРѕ РЅР°РІС‡РµРЅРѕС— РјРѕРґРµР»С– ResNet50
    """
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    return model

# РЎС‚РІРѕСЂРµРЅРЅСЏ РјРѕРґРµР»С– С‚Р° Р±Р°С‚С‡РµСЂР°
model = load_model()
batcher = DynamicBatcher(model, max_batch_size=16, max_wait_time=0.1)

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
    Р•РЅРґРїРѕС–РЅС‚ РґР»СЏ РїСЂРѕРіРЅРѕР·СѓРІР°РЅРЅСЏ Р· РІРёРєРѕСЂРёСЃС‚Р°РЅРЅСЏРј РґРёРЅР°РјС–С‡РЅРѕРіРѕ Р±Р°С‚С‡РёРЅРіСѓ
    """
    if 'file' not in request.files:
        return jsonify({'error': 'РќРµРјР°С” С„Р°Р№Р»Сѓ РІ Р·Р°РїРёС‚С–'}), 400

    file = request.files['file']
    img_bytes = file.read()

    try:
        # РџРѕРїРµСЂРµРґРЅСЏ РѕР±СЂРѕР±РєР° Р·РѕР±СЂР°Р¶РµРЅРЅСЏ
        img = Image.open(io.BytesIO(img_bytes))
        img_tensor = preprocessing(img)

        # Р’РёРєРѕСЂРёСЃС‚Р°РЅРЅСЏ РґРёРЅР°РјС–С‡РЅРѕРіРѕ Р±Р°С‚С‡РёРЅРіСѓ РґР»СЏ РїСЂРѕРіРЅРѕР·СѓРІР°РЅРЅСЏ
        outputs = batcher.predict(img_tensor)

        # РћР±СЂРѕР±РєР° СЂРµР·СѓР»СЊС‚Р°С‚С–РІ
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
    Р•РЅРґРїРѕС–РЅС‚ РґР»СЏ РїРµСЂРµРІС–СЂРєРё СЃС‚Р°РЅСѓ СЃРµСЂРІРµСЂР°
    """
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)

