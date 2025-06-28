# новлена версія для PR
#!/usr/bin/env python

'''
РљРІР°РЅС‚РёР·Р°С†С–СЏ РјРѕРґРµР»С– РґР»СЏ РїСЂРёС€РІРёРґС€РµРЅРЅСЏ С–РЅС„РµСЂРµРЅСЃСѓ
'''

import os
import time
import logging
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.quantization
from torch.quantization import quantize_dynamic, quantize_static, QuantStub, DeQuantStub
from torch.utils.data import DataLoader, Dataset

import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights

# РќР°Р»Р°С€С‚СѓРІР°РЅРЅСЏ Р»РѕРіСѓРІР°РЅРЅСЏ
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('model_quantization')

class QuantizableResNet(nn.Module):
    '''
    РћР±РіРѕСЂС‚РєР° РґР»СЏ ResNet Р· РїС–РґС‚СЂРёРјРєРѕСЋ РєРІР°РЅС‚РёР·Р°С†С–С—
    '''
    def __init__(self, model):
        super(QuantizableResNet, self).__init__()
        self.quant = QuantStub()
        self.model = model
        self.dequant = DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.model(x)
        x = self.dequant(x)
        return x

    def fuse_model(self):
        '''
        Р—Р»РёС‚С‚СЏ РѕРїРµСЂР°С†С–Р№ Conv+BN+ReLU РґР»СЏ РїСЂРёС€РІРёРґС€РµРЅРЅСЏ С–РЅС„РµСЂРµРЅСЃСѓ
        '''
        for module_name, module in self.model.named_children():
            if "layer" in module_name:
                for basic_block_name, basic_block in module.named_children():
                    torch.quantization.fuse_modules(
                        basic_block, ["conv1", "bn1", "relu"], inplace=True
                    )
                    torch.quantization.fuse_modules(
                        basic_block, ["conv2", "bn2"], inplace=True
                    )

        logger.info("РњРѕРґРµР»СЊ СѓСЃРїС–С€РЅРѕ РїС–РґРіРѕС‚РѕРІР»РµРЅР° РґРѕ РєРІР°РЅС‚РёР·Р°С†С–С—")

def load_model(model_path=None):
    '''
    Р—Р°РІР°РЅС‚Р°Р¶РµРЅРЅСЏ РјРѕРґРµР»С– РґР»СЏ РєРІР°РЅС‚РёР·Р°С†С–С—

    РџР°СЂР°РјРµС‚СЂРё:
    -----------
    model_path: С€Р»СЏС… РґРѕ Р·Р±РµСЂРµР¶РµРЅРѕС— РјРѕРґРµР»С– (СЏРєС‰Рѕ None, РІРёРєРѕСЂРёСЃС‚РѕРІСѓС”С‚СЊСЃСЏ РїРѕРїРµСЂРµРґРЅСЊРѕ РЅР°РІС‡РµРЅР° РјРѕРґРµР»СЊ)

    РџРѕРІРµСЂС‚Р°С”:
    -----------
    РјРѕРґРµР»СЊ
    '''
    if model_path and os.path.exists(model_path):
        logger.info(f"Р—Р°РІР°РЅС‚Р°Р¶РµРЅРЅСЏ РјРѕРґРµР»С– Р· {model_path}")
        model = torch.load(model_path)
    else:
        logger.info("Р—Р°РІР°РЅС‚Р°Р¶РµРЅРЅСЏ РїРѕРїРµСЂРµРґРЅСЊРѕ РЅР°РІС‡РµРЅРѕС— РјРѕРґРµР»С– ResNet50")
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

    return model

def get_dataloaders(data_dir='./data', batch_size=64):
    '''
    РЎС‚РІРѕСЂРµРЅРЅСЏ Р·Р°РІР°РЅС‚Р°Р¶СѓРІР°С‡С–РІ РґР°РЅРёС… РґР»СЏ РЅР°РІС‡Р°РЅРЅСЏ С‚Р° РІР°Р»С–РґР°С†С–С—

    РџР°СЂР°РјРµС‚СЂРё:
    -----------
    data_dir: РґРёСЂРµРєС‚РѕСЂС–СЏ Р· РґР°РЅРёРјРё
    batch_size: СЂРѕР·РјС–СЂ Р±Р°С‚С‡Сѓ

    РџРѕРІРµСЂС‚Р°С”:
    -----------
    train_loader, val_loader
    '''
    # РўСЂР°РЅСЃС„РѕСЂРјР°С†С–С— РґР»СЏ РЅР°РІС‡Р°РЅРЅСЏ С‚Р° РІР°Р»С–РґР°С†С–С—
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # РўСЂР°РЅСЃС„РѕСЂРјР°С†С–С— РґР»СЏ РєРІР°РЅС‚РёР·Р°С†С–С— (Р±РµР· РЅРѕСЂРјР°Р»С–Р·Р°С†С–С— РґР»СЏ int8)
    quantization_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])

    # РЎС‚РІРѕСЂРµРЅРЅСЏ РґР°С‚Р°СЃРµС‚С–РІ
    try:
        train_dataset = torchvision.datasets.ImageFolder(
            os.path.join(data_dir, 'train'),
            transform=train_transform
        )

        val_dataset = torchvision.datasets.ImageFolder(
            os.path.join(data_dir, 'val'),
            transform=val_transform
        )

        calib_dataset = torchvision.datasets.ImageFolder(
            os.path.join(data_dir, 'val'),
            transform=quantization_transform
        )
    except Exception as e:
        logger.warning(f"РџРѕРјРёР»РєР° Р·Р°РІР°РЅС‚Р°Р¶РµРЅРЅСЏ РґР°РЅРёС…: {e}. Р’РёРєРѕСЂРёСЃС‚Р°РЅРЅСЏ CIFAR-10 РґР»СЏ РґРµРјРѕРЅСЃС‚СЂР°С†С–С—.")

        # Р—Р°РїР°СЃРЅРёР№ РІР°СЂС–Р°РЅС‚: CIFAR-10
        train_dataset = torchvision.datasets.CIFAR10(
            root='./data',
            train=True,
            download=True,
            transform=train_transform
        )

        val_dataset = torchvision.datasets.CIFAR10(
            root='./data',
            train=False,
            download=True,
            transform=val_transform
        )

        calib_dataset = torchvision.datasets.CIFAR10(
            root='./data',
            train=False,
            download=True,
            transform=quantization_transform
        )

    # РЎС‚РІРѕСЂРµРЅРЅСЏ Р·Р°РІР°РЅС‚Р°Р¶СѓРІР°С‡С–РІ РґР°РЅРёС…
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # РЎС‚РІРѕСЂРµРЅРЅСЏ Р·Р°РІР°РЅС‚Р°Р¶СѓРІР°С‡Р° РґР»СЏ РєР°Р»С–Р±СЂСѓРІР°РЅРЅСЏ (РјРµРЅС€РёР№ СЂРѕР·РјС–СЂ)
    calib_loader = DataLoader(
        calib_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    return train_loader, val_loader, calib_loader

def evaluate_model(model, val_loader, device):
    '''
    РћС†С–РЅРєР° РјРѕРґРµР»С– РЅР° РІР°Р»С–РґР°С†С–Р№РЅРѕРјСѓ РЅР°Р±РѕСЂС– РґР°РЅРёС…

    РџР°СЂР°РјРµС‚СЂРё:
    -----------
    model: РјРѕРґРµР»СЊ РґР»СЏ РѕС†С–РЅРєРё
    val_loader: Р·Р°РІР°РЅС‚Р°Р¶СѓРІР°С‡ РґР°РЅРёС… РґР»СЏ РІР°Р»С–РґР°С†С–С—
    device: РїСЂРёСЃС‚СЂС–Р№ РґР»СЏ РѕР±С‡РёСЃР»РµРЅСЊ

    РџРѕРІРµСЂС‚Р°С”:
    -----------
    С‚РѕС‡РЅС–СЃС‚СЊ РјРѕРґРµР»С–
    '''
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="РћС†С–РЅРєР° РјРѕРґРµР»С–"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    logger.info(f"РўРѕС‡РЅС–СЃС‚СЊ РЅР° РІР°Р»С–РґР°С†С–Р№РЅРѕРјСѓ РЅР°Р±РѕСЂС–: {accuracy:.2f}%")

    return accuracy

def measure_inference_time(model, val_loader, device, num_runs=100):
    '''
    Р’РёРјС–СЂСЋРІР°РЅРЅСЏ С‡Р°СЃСѓ С–РЅС„РµСЂРµРЅСЃСѓ РјРѕРґРµР»С–

    РџР°СЂР°РјРµС‚СЂРё:
    -----------
    model: РјРѕРґРµР»СЊ РґР»СЏ РІРёРјС–СЂСЋРІР°РЅРЅСЏ
    val_loader: Р·Р°РІР°РЅС‚Р°Р¶СѓРІР°С‡ РґР°РЅРёС… РґР»СЏ РІР°Р»С–РґР°С†С–С—
    device: РїСЂРёСЃС‚СЂС–Р№ РґР»СЏ РѕР±С‡РёСЃР»РµРЅСЊ
    num_runs: РєС–Р»СЊРєС–СЃС‚СЊ Р·Р°РїСѓСЃРєС–РІ РґР»СЏ РІРёРјС–СЂСЋРІР°РЅРЅСЏ

    РџРѕРІРµСЂС‚Р°С”:
    -----------
    СЃРµСЂРµРґРЅС–Р№ С‡Р°СЃ С–РЅС„РµСЂРµРЅСЃСѓ
    '''
    model.eval()
    times = []

    # РћС‚СЂРёРјР°РЅРЅСЏ РїРµСЂС€РѕРіРѕ Р±Р°С‚С‡Сѓ РґР»СЏ РІРёРјС–СЂСЋРІР°РЅРЅСЏ
    for inputs, _ in val_loader:
        inputs = inputs.to(device)
        break

    # Р РѕР·С–РіСЂС–РІ
    with torch.no_grad():
        for _ in range(10):
            _ = model(inputs)

    # Р’РёРјС–СЂСЋРІР°РЅРЅСЏ С‡Р°СЃСѓ
    with torch.no_grad():
        for _ in range(num_runs):
            start_time = time.time()
            _ = model(inputs)
            end_time = time.time()
            times.append(end_time - start_time)

    # РћР±С‡РёСЃР»РµРЅРЅСЏ СЃС‚Р°С‚РёСЃС‚РёРєРё
    avg_time = sum(times) / len(times)
    std_time = np.std(times)

    logger.info(f"РЎРµСЂРµРґРЅС–Р№ С‡Р°СЃ С–РЅС„РµСЂРµРЅСЃСѓ: {avg_time*1000:.2f} РјСЃ В± {std_time*1000:.2f} РјСЃ")

    return avg_time

def get_model_size(model):
    '''
    РћС‚СЂРёРјР°РЅРЅСЏ СЂРѕР·РјС–СЂСѓ РјРѕРґРµР»С– РІ РњР‘

    РџР°СЂР°РјРµС‚СЂРё:
    -----------
    model: РјРѕРґРµР»СЊ РґР»СЏ РІРёРјС–СЂСЋРІР°РЅРЅСЏ

    РџРѕРІРµСЂС‚Р°С”:
    -----------
    СЂРѕР·РјС–СЂ РјРѕРґРµР»С– РІ РњР‘
    '''
    torch.save(model.state_dict(), "temp_model.pt")
    size_mb = os.path.getsize("temp_model.pt") / (1024 * 1024)
    os.remove("temp_model.pt")
    return size_mb

def calibrate_model(model, calib_loader, device):
    '''
    РљР°Р»С–Р±СЂСѓРІР°РЅРЅСЏ РјРѕРґРµР»С– РґР»СЏ СЃС‚Р°С‚РёС‡РЅРѕС— РєРІР°РЅС‚РёР·Р°С†С–С—

    РџР°СЂР°РјРµС‚СЂРё:
    -----------
    model: РјРѕРґРµР»СЊ РґР»СЏ РєР°Р»С–Р±СЂСѓРІР°РЅРЅСЏ
    calib_loader: Р·Р°РІР°РЅС‚Р°Р¶СѓРІР°С‡ РґР°РЅРёС… РґР»СЏ РєР°Р»С–Р±СЂСѓРІР°РЅРЅСЏ
    device: РїСЂРёСЃС‚СЂС–Р№ РґР»СЏ РѕР±С‡РёСЃР»РµРЅСЊ

    РџРѕРІРµСЂС‚Р°С”:
    -----------
    РІС–РґРєР°Р»С–Р±СЂРѕРІР°РЅР° РјРѕРґРµР»СЊ
    '''
    logger.info("РљР°Р»С–Р±СЂСѓРІР°РЅРЅСЏ РјРѕРґРµР»С– РґР»СЏ СЃС‚Р°С‚РёС‡РЅРѕС— РєРІР°РЅС‚РёР·Р°С†С–С—")
    model.eval()

    # РќР°Р»Р°С€С‚СѓРІР°РЅРЅСЏ РєР°Р»С–Р±СЂСѓРІР°РЅРЅСЏ
    with torch.no_grad():
        # РћР±РјРµР¶СѓС”РјРѕСЃСЊ РјРµРЅС€РѕСЋ РєС–Р»СЊРєС–СЃС‚СЋ Р±Р°С‚С‡С–РІ РґР»СЏ С€РІРёРґС€РѕРіРѕ РєР°Р»С–Р±СЂСѓРІР°РЅРЅСЏ
        for i, (inputs, _) in enumerate(tqdm(calib_loader, desc="РљР°Р»С–Р±СЂСѓРІР°РЅРЅСЏ")):
            inputs = inputs.to(device)
            model(inputs)
            if i >= 10:  # Р—Р°Р·РІРёС‡Р°Р№ РґРѕСЃС‚Р°С‚РЅСЊРѕ 10-20 Р±Р°С‚С‡С–РІ
                break

    return model

def dynamic_quantize_model(model):
    '''
    Р”РёРЅР°РјС–С‡РЅР° РєРІР°РЅС‚РёР·Р°С†С–СЏ РјРѕРґРµР»С–

    РџР°СЂР°РјРµС‚СЂРё:
    -----------
    model: РјРѕРґРµР»СЊ РґР»СЏ РєРІР°РЅС‚РёР·Р°С†С–С—

    РџРѕРІРµСЂС‚Р°С”:
    -----------
    РєРІР°РЅС‚РёР·РѕРІР°РЅР° РјРѕРґРµР»СЊ
    '''
    logger.info("Р—Р°СЃС‚РѕСЃСѓРІР°РЅРЅСЏ РґРёРЅР°РјС–С‡РЅРѕС— РєРІР°РЅС‚РёР·Р°С†С–С—")

    # Р”РёРЅР°РјС–С‡РЅР° РєРІР°РЅС‚РёР·Р°С†С–СЏ Р»С–РЅС–Р№РЅРёС… С€Р°СЂС–РІ
    quantized_model = torch.quantization.quantize_dynamic(
        model, 
        {nn.Linear}, 
        dtype=torch.qint8
    )

    return quantized_model

def static_quantize_model(model, calib_loader):
    '''
    РЎС‚Р°С‚РёС‡РЅР° РєРІР°РЅС‚РёР·Р°С†С–СЏ РјРѕРґРµР»С–

    РџР°СЂР°РјРµС‚СЂРё:
    -----------
    model: РјРѕРґРµР»СЊ РґР»СЏ РєРІР°РЅС‚РёР·Р°С†С–С—
    calib_loader: Р·Р°РІР°РЅС‚Р°Р¶СѓРІР°С‡ РґР°РЅРёС… РґР»СЏ РєР°Р»С–Р±СЂСѓРІР°РЅРЅСЏ

    РџРѕРІРµСЂС‚Р°С”:
    -----------
    РєРІР°РЅС‚РёР·РѕРІР°РЅР° РјРѕРґРµР»СЊ
    '''
    logger.info("РџС–РґРіРѕС‚РѕРІРєР° РґРѕ СЃС‚Р°С‚РёС‡РЅРѕС— РєРІР°РЅС‚РёР·Р°С†С–С—")

    # РџС–РґРіРѕС‚РѕРІРєР° РјРѕРґРµР»С– Р· QuantStub С‚Р° DeQuantStub
    quantizable_model = QuantizableResNet(model)

    # Р—Р»РёС‚С‚СЏ С€Р°СЂС–РІ Conv+BN+ReLU
    quantizable_model.fuse_model()

    # РџРµСЂРµРІРµРґРµРЅРЅСЏ РјРѕРґРµР»С– РІ СЂРµР¶РёРј РѕС†С–РЅРєРё
    quantizable_model.eval()

    # Р’СЃС‚Р°РЅРѕРІР»РµРЅРЅСЏ РєРѕРЅС„С–РіСѓСЂР°С†С–С— РєРІР°РЅС‚РёР·Р°С†С–С—
    quantizable_model.qconfig = torch.quantization.get_default_qconfig('fbgemm')

    # РџС–РґРіРѕС‚РѕРІРєР° РјРѕРґРµР»С– РґРѕ РєРІР°РЅС‚РёР·Р°С†С–С—
    torch.quantization.prepare(quantizable_model, inplace=True)

    # РљР°Р»С–Р±СЂСѓРІР°РЅРЅСЏ РјРѕРґРµР»С–
    with torch.no_grad():
        for i, (inputs, _) in enumerate(tqdm(calib_loader, desc="РљР°Р»С–Р±СЂСѓРІР°РЅРЅСЏ")):
            quantizable_model(inputs)
            if i >= 10:  # РћР±РјРµР¶СѓС”РјРѕСЃСЊ РґР»СЏ С€РІРёРґС€РѕРіРѕ РєР°Р»С–Р±СЂСѓРІР°РЅРЅСЏ
                break

    # РљРѕРЅРІРµСЂС‚Р°С†С–СЏ РІ РєРІР°РЅС‚РёР·РѕРІР°РЅСѓ РјРѕРґРµР»СЊ
    torch.quantization.convert(quantizable_model, inplace=True)

    logger.info("РЎС‚Р°С‚РёС‡РЅР° РєРІР°РЅС‚РёР·Р°С†С–СЏ Р·Р°РІРµСЂС€РµРЅР°")

    return quantizable_model

def main():
    parser = argparse.ArgumentParser(description="РљРІР°РЅС‚РёР·Р°С†С–СЏ РјРѕРґРµР»С– РґР»СЏ РїСЂРёС€РІРёРґС€РµРЅРЅСЏ С–РЅС„РµСЂРµРЅСЃСѓ")
    parser.add_argument("--model", type=str, default=None,
                        help="РЁР»СЏС… РґРѕ РјРѕРґРµР»С– (СЏРєС‰Рѕ None, РІРёРєРѕСЂРёСЃС‚РѕРІСѓС”С‚СЊСЃСЏ РїРѕРїРµСЂРµРґРЅСЊРѕ РЅР°РІС‡РµРЅР° РјРѕРґРµР»СЊ)")
    parser.add_argument("--data-dir", type=str, default="./data",
                        help="Р”РёСЂРµРєС‚РѕСЂС–СЏ Р· РґР°РЅРёРјРё")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Р РѕР·РјС–СЂ Р±Р°С‚С‡Сѓ")
    parser.add_argument("--quantization-method", type=str, choices=["dynamic", "static"], default="dynamic",
                        help="РњРµС‚РѕРґ РєРІР°РЅС‚РёР·Р°С†С–С— (dynamic Р°Р±Рѕ static)")
    parser.add_argument("--save-path", type=str, default="quantized_model.pt",
                        help="РЁР»СЏС… РґР»СЏ Р·Р±РµСЂРµР¶РµРЅРЅСЏ РјРѕРґРµР»С–")

    args = parser.parse_args()

    # Р—Р°РІР°РЅС‚Р°Р¶РµРЅРЅСЏ РјРѕРґРµР»С–
    model = load_model(args.model)

    # Р’РёР·РЅР°С‡РµРЅРЅСЏ РїСЂРёСЃС‚СЂРѕСЋ РґР»СЏ РїРѕС‡Р°С‚РєРѕРІРѕС— РѕС†С–РЅРєРё
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # РћС‚СЂРёРјР°РЅРЅСЏ Р·Р°РІР°РЅС‚Р°Р¶СѓРІР°С‡С–РІ РґР°РЅРёС…
    train_loader, val_loader, calib_loader = get_dataloaders(args.data_dir, args.batch_size)

    # РћС†С–РЅРєР° РјРѕРґРµР»С– РїРµСЂРµРґ РєРІР°РЅС‚РёР·Р°С†С–С”СЋ
    logger.info("РћС†С–РЅРєР° РјРѕРґРµР»С– РїРµСЂРµРґ РєРІР°РЅС‚РёР·Р°С†С–С”СЋ")
    initial_size = get_model_size(model)
    initial_accuracy = evaluate_model(model, val_loader, device)
    initial_inference_time = measure_inference_time(model, val_loader, device)

    # РљРІР°РЅС‚РёР·Р°С†С–СЏ РІРёРєРѕРЅСѓС”С‚СЊСЃСЏ РЅР° CPU
    cpu_device = torch.device("cpu")
    model = model.to(cpu_device)

    # РљРІР°РЅС‚РёР·Р°С†С–СЏ РјРѕРґРµР»С–
    if args.quantization_method == "dynamic":
        quantized_model = dynamic_quantize_model(model)
    else:  # static
        quantized_model = static_quantize_model(model, calib_loader)

    # РћС†С–РЅРєР° РєРІР°РЅС‚РёР·РѕРІР°РЅРѕС— РјРѕРґРµР»С–
    logger.info("РћС†С–РЅРєР° РєРІР°РЅС‚РёР·РѕРІР°РЅРѕС— РјРѕРґРµР»С–")
    quantized_size = get_model_size(quantized_model)
    quantized_accuracy = evaluate_model(quantized_model, val_loader, cpu_device)
    quantized_inference_time = measure_inference_time(quantized_model, val_loader, cpu_device)

    # Р—Р±РµСЂРµР¶РµРЅРЅСЏ РјРѕРґРµР»С–
    torch.save(quantized_model, args.save_path)
    logger.info(f"РљРІР°РЅС‚РёР·РѕРІР°РЅР° РјРѕРґРµР»СЊ Р·Р±РµСЂРµР¶РµРЅР° РІ {args.save_path}")

    # Р’РёРІРµРґРµРЅРЅСЏ СЂРµР·СѓР»СЊС‚Р°С‚С–РІ
    logger.info("\nР РµР·СѓР»СЊС‚Р°С‚Рё РєРІР°РЅС‚РёР·Р°С†С–С—:")
    logger.info(f"РџРѕС‡Р°С‚РєРѕРІРёР№ СЂРѕР·РјС–СЂ РјРѕРґРµР»С–: {initial_size:.2f} РњР‘")
    logger.info(f"РџРѕС‡Р°С‚РєРѕРІР° С‚РѕС‡РЅС–СЃС‚СЊ: {initial_accuracy:.2f}%")
    logger.info(f"РџРѕС‡Р°С‚РєРѕРІРёР№ С‡Р°СЃ С–РЅС„РµСЂРµРЅСЃСѓ: {initial_inference_time*1000:.2f} РјСЃ")

    logger.info(f"\nР РѕР·РјС–СЂ РєРІР°РЅС‚РёР·РѕРІР°РЅРѕС— РјРѕРґРµР»С–: {quantized_size:.2f} РњР‘ ({quantized_size/initial_size*100:.2f}% РІС–Рґ РїРѕС‡Р°С‚РєРѕРІРѕС—)")
    logger.info(f"РўРѕС‡РЅС–СЃС‚СЊ РєРІР°РЅС‚РёР·РѕРІР°РЅРѕС— РјРѕРґРµР»С–: {quantized_accuracy:.2f}% ({quantized_accuracy-initial_accuracy:+.2f}%)")
    logger.info(f"Р§Р°СЃ С–РЅС„РµСЂРµРЅСЃСѓ РєРІР°РЅС‚РёР·РѕРІР°РЅРѕС— РјРѕРґРµР»С–: {quantized_inference_time*1000:.2f} РјСЃ ({(1-quantized_inference_time/initial_inference_time)*100:.2f}% РїСЂРёС€РІРёРґС€РµРЅРЅСЏ)")

if __name__ == "__main__":
    main()

