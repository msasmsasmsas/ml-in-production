# Updated version for PR
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

# Р“Р»РѕР±Р°Р»СЊРЅС– Р·РјС–РЅРЅС– РґР»СЏ РјРѕРґРµР»С– С‚Р° С‚СЂР°РЅСЃС„РѕСЂРјР°С†С–Р№
model = None
preprocessing = None
labels = None

def load_model(device=None):
    """
    Р—Р°РІР°РЅС‚Р°Р¶РµРЅРЅСЏ РїРѕРїРµСЂРµРґРЅСЊРѕ РЅР°РІС‡РµРЅРѕС— РјРѕРґРµР»С– ResNet50

    РџР°СЂР°РјРµС‚СЂРё:
    -----------
    device: РїСЂРёСЃС‚СЂС–Р№ РґР»СЏ РІРёРєРѕРЅР°РЅРЅСЏ (cuda Р°Р±Рѕ cpu)

    РџРѕРІРµСЂС‚Р°С”:
    -----------
    Р·Р°РІР°РЅС‚Р°Р¶РµРЅР° РјРѕРґРµР»СЊ
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)

    print(f"Р—Р°РІР°РЅС‚Р°Р¶РµРЅРЅСЏ РјРѕРґРµР»С– РЅР° РїСЂРёСЃС‚СЂС–Р№: {device}")

    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    model.to(device)
    model.eval()

    return model

def setup():
    """
    Р†РЅС–С†С–Р°Р»С–Р·Р°С†С–СЏ РјРѕРґРµР»С–, С‚СЂР°РЅСЃС„РѕСЂРјР°С†С–Р№ С‚Р° РєР»Р°СЃС–РІ
    """
    global model, preprocessing, labels

    # Р—Р°РІР°РЅС‚Р°Р¶РµРЅРЅСЏ РјРѕРґРµР»С–
    model = load_model()

    # РЎС‚РІРѕСЂРµРЅРЅСЏ С‚СЂР°РЅСЃС„РѕСЂРјР°С†С–Р№ РґР»СЏ Р·РѕР±СЂР°Р¶РµРЅСЊ
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
    Р•РЅРґРїРѕС–РЅС‚ РґР»СЏ РїСЂРѕРіРЅРѕР·СѓРІР°РЅРЅСЏ Р·РѕР±СЂР°Р¶РµРЅРЅСЏ

    РџРѕРІРµСЂС‚Р°С”:
    -----------
    JSON Р· СЂРµР·СѓР»СЊС‚Р°С‚Р°РјРё РїСЂРѕРіРЅРѕР·СѓРІР°РЅРЅСЏ
    """
    request_id = str(uuid.uuid4())
    start_time = time.time()

    try:
        # РџРµСЂРµРІС–СЂРєР° РЅР°СЏРІРЅРѕСЃС‚С– С„Р°Р№Р»Сѓ РІ Р·Р°РїРёС‚С–
        if 'file' not in request.files:
            return jsonify({
                'request_id': request_id,
                'success': False,
                'error': 'Р¤Р°Р№Р» РЅРµ Р·РЅР°Р№РґРµРЅРѕ РІ Р·Р°РїРёС‚С–',
                'processing_time': (time.time() - start_time) * 1000
            }), 400

        # РћС‚СЂРёРјР°РЅРЅСЏ С„Р°Р№Р»Сѓ
        file = request.files['file']
        img_bytes = file.read()

        # Р—Р°РІР°РЅС‚Р°Р¶РµРЅРЅСЏ С‚Р° РїРѕРїРµСЂРµРґРЅСЏ РѕР±СЂРѕР±РєР° Р·РѕР±СЂР°Р¶РµРЅРЅСЏ
        img = Image.open(io.BytesIO(img_bytes))
        img_tensor = preprocessing(img).unsqueeze(0)  # РґРѕРґР°РІР°РЅРЅСЏ РІРёРјС–СЂРЅРѕСЃС‚С– Р±Р°С‚С‡Сѓ

        # РџРµСЂРµРјС–С‰РµРЅРЅСЏ С‚РµРЅР·РѕСЂР° РЅР° GPU, СЏРєС‰Рѕ РґРѕСЃС‚СѓРїРЅРѕ
        device = next(model.parameters()).device
        img_tensor = img_tensor.to(device)

        # РџСЂРѕРіРЅРѕР·СѓРІР°РЅРЅСЏ
        with torch.no_grad():
            outputs = model(img_tensor)

        # РћР±СЂРѕР±РєР° СЂРµР·СѓР»СЊС‚Р°С‚С–РІ
        probs = torch.nn.functional.softmax(outputs, dim=1)[0]
        top5_probs, top5_indices = torch.topk(probs, 5)

        # РџС–РґРіРѕС‚РѕРІРєР° РІС–РґРїРѕРІС–РґС–
        predictions = []
        for i, (idx, prob) in enumerate(zip(top5_indices.cpu().numpy(), top5_probs.cpu().numpy())):
            predictions.append({
                'class_id': int(idx),
                'class_name': labels.get(str(idx), f"Unknown class {idx}"),
                'score': float(prob)
            })

        # Р§Р°СЃ РѕР±СЂРѕР±РєРё РІ РјС–Р»С–СЃРµРєСѓРЅРґР°С…
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
        # РћР±СЂРѕР±РєР° РїРѕРјРёР»РѕРє
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
    Р•РЅРґРїРѕС–РЅС‚ РґР»СЏ РїРµСЂРµРІС–СЂРєРё СЃС‚Р°РЅСѓ СЃРµСЂРІРµСЂР°

    РџРѕРІРµСЂС‚Р°С”:
    -----------
    JSON Р·С– СЃС‚Р°С‚СѓСЃРѕРј СЃРµСЂРІРµСЂР°
    """
    # РџРµСЂРµРІС–СЂРєР° СЃС‚Р°РЅСѓ РјРѕРґРµР»С–
    if model is None:
        return jsonify({
            'status': 'not_ready',
            'error': 'РњРѕРґРµР»СЊ РЅРµ Р·Р°РІР°РЅС‚Р°Р¶РµРЅР°'
        }), 503

    # РћС‚СЂРёРјР°РЅРЅСЏ С–РЅС„РѕСЂРјР°С†С–С— РїСЂРѕ РїСЂРёСЃС‚СЂС–Р№
    device = next(model.parameters()).device

    return jsonify({
        'status': 'ok',
        'model': 'ResNet50',
        'framework': 'PyTorch',
        'device': device.type,
        'version': '1.0.0'
    })

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='REST СЃРµСЂРІРµСЂ РґР»СЏ С–РЅС„РµСЂРµРЅСЃСѓ РјРѕРґРµР»РµР№')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='РҐРѕСЃС‚ РґР»СЏ РїСЂРѕСЃР»СѓС…РѕРІСѓРІР°РЅРЅСЏ')
    parser.add_argument('--port', type=int, default=5000, help='РџРѕСЂС‚ РґР»СЏ РїСЂРѕСЃР»СѓС…РѕРІСѓРІР°РЅРЅСЏ')
    parser.add_argument('--debug', action='store_true', help='Р РµР¶РёРј РІС–РґР»Р°РіРѕРґР¶РµРЅРЅСЏ Flask')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], default=None, 
                        help='РџСЂРёСЃС‚СЂС–Р№ РґР»СЏ РІРёРєРѕРЅР°РЅРЅСЏ (Р·Р° Р·Р°РјРѕРІС‡СѓРІР°РЅРЅСЏРј: Р°РІС‚РѕРІРёР·РЅР°С‡РµРЅРЅСЏ)')

    args = parser.parse_args()

    # Р†РЅС–С†С–Р°Р»С–Р·Р°С†С–СЏ РјРѕРґРµР»С– С‚Р° С–РЅС€РёС… РєРѕРјРїРѕРЅРµРЅС‚С–РІ
    setup()

    if args.debug:
        # Р—Р°РїСѓСЃРє Сѓ СЂРµР¶РёРјС– РІС–РґР»Р°РіРѕРґР¶РµРЅРЅСЏ (РЅРµ СЂРµРєРѕРјРµРЅРґСѓС”С‚СЊСЃСЏ РґР»СЏ РїСЂРѕРґР°РєС€РµРЅСѓ)
        app.run(host=args.host, port=args.port, debug=True)
    else:
        # Р—Р°РїСѓСЃРє С‡РµСЂРµР· Waitress РґР»СЏ РєСЂР°С‰РѕС— РїСЂРѕРґСѓРєС‚РёРІРЅРѕСЃС‚С–
        print(f"Р—Р°РїСѓСЃРє СЃРµСЂРІРµСЂР° РЅР° {args.host}:{args.port}")
        serve(app, host=args.host, port=args.port, threads=8)

