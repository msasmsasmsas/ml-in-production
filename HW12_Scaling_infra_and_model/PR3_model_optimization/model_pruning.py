# новлена версія для PR
#!/usr/bin/env python

'''
РџСЂРѕСЂС–РґР¶СѓРІР°РЅРЅСЏ РјРѕРґРµР»С– РґР»СЏ РїСЂРёС€РІРёРґС€РµРЅРЅСЏ С–РЅС„РµСЂРµРЅСЃСѓ
'''

import os
import time
import logging
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.utils.prune as prune
from torch.utils.data import DataLoader, Dataset

import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights

# РќР°Р»Р°С€С‚СѓРІР°РЅРЅСЏ Р»РѕРіСѓРІР°РЅРЅСЏ
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('model_pruning')

def load_model(model_path=None):
    '''
    Р—Р°РІР°РЅС‚Р°Р¶РµРЅРЅСЏ РјРѕРґРµР»С– РґР»СЏ РїСЂРѕСЂС–РґР¶СѓРІР°РЅРЅСЏ

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

    return train_loader, val_loader

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

def count_parameters(model):
    '''
    РџС–РґСЂР°С…СѓРЅРѕРє РєС–Р»СЊРєРѕСЃС‚С– РїР°СЂР°РјРµС‚СЂС–РІ РјРѕРґРµР»С–

    РџР°СЂР°РјРµС‚СЂРё:
    -----------
    model: РјРѕРґРµР»СЊ РґР»СЏ РїС–РґСЂР°С…СѓРЅРєСѓ

    РџРѕРІРµСЂС‚Р°С”:
    -----------
    Р·Р°РіР°Р»СЊРЅР° РєС–Р»СЊРєС–СЃС‚СЊ РїР°СЂР°РјРµС‚СЂС–РІ
    '''
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def apply_global_pruning(model, amount):
    '''
    Р—Р°СЃС‚РѕСЃСѓРІР°РЅРЅСЏ РіР»РѕР±Р°Р»СЊРЅРѕРіРѕ РїСЂРѕСЂС–РґР¶СѓРІР°РЅРЅСЏ РґРѕ РјРѕРґРµР»С–

    РџР°СЂР°РјРµС‚СЂРё:
    -----------
    model: РјРѕРґРµР»СЊ РґР»СЏ РїСЂРѕСЂС–РґР¶СѓРІР°РЅРЅСЏ
    amount: РІС–РґСЃРѕС‚РѕРє РІР°Рі РґР»СЏ РІРёРґР°Р»РµРЅРЅСЏ (0-1)

    РџРѕРІРµСЂС‚Р°С”:
    -----------
    РїСЂРѕСЂС–РґР¶РµРЅР° РјРѕРґРµР»СЊ
    '''
    logger.info(f"Р—Р°СЃС‚РѕСЃСѓРІР°РЅРЅСЏ РіР»РѕР±Р°Р»СЊРЅРѕРіРѕ РїСЂРѕСЂС–РґР¶СѓРІР°РЅРЅСЏ Р· РєРѕРµС„С–С†С–С”РЅС‚РѕРј {amount}")

    # РћС‚СЂРёРјР°РЅРЅСЏ РІСЃС–С… РїР°СЂР°РјРµС‚СЂС–РІ РґР»СЏ РїСЂРѕСЂС–РґР¶СѓРІР°РЅРЅСЏ
    parameters_to_prune = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            parameters_to_prune.append((module, 'weight'))

    # Р—Р°СЃС‚РѕСЃСѓРІР°РЅРЅСЏ РіР»РѕР±Р°Р»СЊРЅРѕРіРѕ РїСЂРѕСЂС–РґР¶СѓРІР°РЅРЅСЏ
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=amount,
    )

    return model

def apply_local_pruning(model, amount):
    '''
    Р—Р°СЃС‚РѕСЃСѓРІР°РЅРЅСЏ Р»РѕРєР°Р»СЊРЅРѕРіРѕ РїСЂРѕСЂС–РґР¶СѓРІР°РЅРЅСЏ РґРѕ РјРѕРґРµР»С–

    РџР°СЂР°РјРµС‚СЂРё:
    -----------
    model: РјРѕРґРµР»СЊ РґР»СЏ РїСЂРѕСЂС–РґР¶СѓРІР°РЅРЅСЏ
    amount: РІС–РґСЃРѕС‚РѕРє РІР°Рі РґР»СЏ РІРёРґР°Р»РµРЅРЅСЏ (0-1)

    РџРѕРІРµСЂС‚Р°С”:
    -----------
    РїСЂРѕСЂС–РґР¶РµРЅР° РјРѕРґРµР»СЊ
    '''
    logger.info(f"Р—Р°СЃС‚РѕСЃСѓРІР°РЅРЅСЏ Р»РѕРєР°Р»СЊРЅРѕРіРѕ РїСЂРѕСЂС–РґР¶СѓРІР°РЅРЅСЏ Р· РєРѕРµС„С–С†С–С”РЅС‚РѕРј {amount}")

    # РџСЂРѕСЂС–РґР¶СѓРІР°РЅРЅСЏ С€Р°СЂС–РІ
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            prune.l1_unstructured(module, name='weight', amount=amount)
        elif isinstance(module, nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=amount)

    return model

def fine_tune_pruned_model(model, train_loader, val_loader, num_epochs=5, learning_rate=0.001):
    '''
    Р”РѕРЅР°РІС‡Р°РЅРЅСЏ РїСЂРѕСЂС–РґР¶СѓРІР°РЅРѕС— РјРѕРґРµР»С–

    РџР°СЂР°РјРµС‚СЂРё:
    -----------
    model: РїСЂРѕСЂС–РґР¶РµРЅР° РјРѕРґРµР»СЊ
    train_loader: Р·Р°РІР°РЅС‚Р°Р¶СѓРІР°С‡ РґР°РЅРёС… РґР»СЏ РЅР°РІС‡Р°РЅРЅСЏ
    val_loader: Р·Р°РІР°РЅС‚Р°Р¶СѓРІР°С‡ РґР°РЅРёС… РґР»СЏ РІР°Р»С–РґР°С†С–С—
    num_epochs: РєС–Р»СЊРєС–СЃС‚СЊ РµРїРѕС… РЅР°РІС‡Р°РЅРЅСЏ
    learning_rate: С€РІРёРґРєС–СЃС‚СЊ РЅР°РІС‡Р°РЅРЅСЏ

    РџРѕРІРµСЂС‚Р°С”:
    -----------
    РґРѕРЅР°РІС‡РµРЅР° РјРѕРґРµР»СЊ
    '''
    logger.info("Р”РѕРЅР°РІС‡Р°РЅРЅСЏ РїСЂРѕСЂС–РґР¶СѓРІР°РЅРѕС— РјРѕРґРµР»С–")

    # Р’РёР·РЅР°С‡РµРЅРЅСЏ РїСЂРёСЃС‚СЂРѕСЋ
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Р’РёР·РЅР°С‡РµРЅРЅСЏ РѕРїС‚РёРјС–Р·Р°С‚РѕСЂР° С‚Р° С„СѓРЅРєС†С–С— РІС‚СЂР°С‚Рё
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # РќР°РІС‡Р°РЅРЅСЏ РјРѕРґРµР»С–
    best_acc = 0.0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        train_bar = tqdm(train_loader, desc=f"Р•РїРѕС…Р° {epoch+1}/{num_epochs}")
        for inputs, labels in train_bar:
            inputs, labels = inputs.to(device), labels.to(device)

            # РћС‡РёС‰РµРЅРЅСЏ РіСЂР°РґС–С”РЅС‚С–РІ
            optimizer.zero_grad()

            # РџСЂСЏРјРёР№ РїСЂРѕС…С–Рґ
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Р—РІРѕСЂРѕС‚РЅС–Р№ РїСЂРѕС…С–Рґ С‚Р° РѕРїС‚РёРјС–Р·Р°С†С–СЏ
            loss.backward()
            optimizer.step()

            # РЎС‚Р°С‚РёСЃС‚РёРєР°
            running_loss += loss.item() * inputs.size(0)
            train_bar.set_postfix({'loss': loss.item()})

        # РћРЅРѕРІР»РµРЅРЅСЏ РїР»Р°РЅСѓРІР°Р»СЊРЅРёРєР°
        scheduler.step()

        # РћС†С–РЅРєР° РјРѕРґРµР»С–
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        logger.info(f"Р•РїРѕС…Р° {epoch+1}/{num_epochs} - Р’С‚СЂР°С‚Р°: {running_loss/len(train_loader.dataset):.4f}, РўРѕС‡РЅС–СЃС‚СЊ: {accuracy:.2f}%")

        # Р—Р±РµСЂРµР¶РµРЅРЅСЏ РЅР°Р№РєСЂР°С‰РѕС— РјРѕРґРµР»С–
        if accuracy > best_acc:
            best_acc = accuracy
            torch.save(model, 'pruned_model_best.pt')

    # Р—Р°РІР°РЅС‚Р°Р¶РµРЅРЅСЏ РЅР°Р№РєСЂР°С‰РѕС— РјРѕРґРµР»С–
    model = torch.load('pruned_model_best.pt')

    return model

def remove_pruning_masks(model):
    '''
    Р’РёРґР°Р»РµРЅРЅСЏ РјР°СЃРѕРє РїСЂРѕСЂС–РґР¶СѓРІР°РЅРЅСЏ РґР»СЏ РїРѕСЃС‚С–Р№РЅРѕРіРѕ РµС„РµРєС‚Сѓ

    РџР°СЂР°РјРµС‚СЂРё:
    -----------
    model: РїСЂРѕСЂС–РґР¶РµРЅР° РјРѕРґРµР»СЊ

    РџРѕРІРµСЂС‚Р°С”:
    -----------
    РјРѕРґРµР»СЊ Р· РІРёРґР°Р»РµРЅРёРјРё РјР°СЃРєР°РјРё РїСЂРѕСЂС–РґР¶СѓРІР°РЅРЅСЏ
    '''
    logger.info("Р’РёРґР°Р»РµРЅРЅСЏ РјР°СЃРѕРє РїСЂРѕСЂС–РґР¶СѓРІР°РЅРЅСЏ")

    # Р’РёРґР°Р»РµРЅРЅСЏ РјР°СЃРѕРє
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            try:
                prune.remove(module, 'weight')
            except:
                pass

    return model

def main():
    parser = argparse.ArgumentParser(description="РџСЂРѕСЂС–РґР¶СѓРІР°РЅРЅСЏ РјРѕРґРµР»С– РґР»СЏ РїСЂРёС€РІРёРґС€РµРЅРЅСЏ С–РЅС„РµСЂРµРЅСЃСѓ")
    parser.add_argument("--model", type=str, default=None,
                        help="РЁР»СЏС… РґРѕ РјРѕРґРµР»С– (СЏРєС‰Рѕ None, РІРёРєРѕСЂРёСЃС‚РѕРІСѓС”С‚СЊСЃСЏ РїРѕРїРµСЂРµРґРЅСЊРѕ РЅР°РІС‡РµРЅР° РјРѕРґРµР»СЊ)")
    parser.add_argument("--data-dir", type=str, default="./data",
                        help="Р”РёСЂРµРєС‚РѕСЂС–СЏ Р· РґР°РЅРёРјРё")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Р РѕР·РјС–СЂ Р±Р°С‚С‡Сѓ")
    parser.add_argument("--pruning-method", type=str, choices=["global", "local"], default="global",
                        help="РњРµС‚РѕРґ РїСЂРѕСЂС–РґР¶СѓРІР°РЅРЅСЏ (global Р°Р±Рѕ local)")
    parser.add_argument("--amount", type=float, default=0.3,
                        help="Р’С–РґСЃРѕС‚РѕРє РІР°Рі РґР»СЏ РІРёРґР°Р»РµРЅРЅСЏ (0-1)")
    parser.add_argument("--fine-tune", action="store_true",
                        help="Р”РѕРЅР°РІС‡Р°РЅРЅСЏ РїС–СЃР»СЏ РїСЂРѕСЂС–РґР¶СѓРІР°РЅРЅСЏ")
    parser.add_argument("--fine-tune-epochs", type=int, default=5,
                        help="РљС–Р»СЊРєС–СЃС‚СЊ РµРїРѕС… РґРѕРЅР°РІС‡Р°РЅРЅСЏ")
    parser.add_argument("--fine-tune-lr", type=float, default=0.001,
                        help="РЁРІРёРґРєС–СЃС‚СЊ РЅР°РІС‡Р°РЅРЅСЏ РґР»СЏ РґРѕРЅР°РІС‡Р°РЅРЅСЏ")
    parser.add_argument("--save-path", type=str, default="pruned_model.pt",
                        help="РЁР»СЏС… РґР»СЏ Р·Р±РµСЂРµР¶РµРЅРЅСЏ РјРѕРґРµР»С–")

    args = parser.parse_args()

    # Р—Р°РІР°РЅС‚Р°Р¶РµРЅРЅСЏ РјРѕРґРµР»С–
    model = load_model(args.model)

    # Р’РёР·РЅР°С‡РµРЅРЅСЏ РїСЂРёСЃС‚СЂРѕСЋ
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # РћС‚СЂРёРјР°РЅРЅСЏ Р·Р°РІР°РЅС‚Р°Р¶СѓРІР°С‡С–РІ РґР°РЅРёС…
    train_loader, val_loader = get_dataloaders(args.data_dir, args.batch_size)

    # РћС†С–РЅРєР° РјРѕРґРµР»С– РїРµСЂРµРґ РїСЂРѕСЂС–РґР¶СѓРІР°РЅРЅСЏРј
    logger.info("РћС†С–РЅРєР° РјРѕРґРµР»С– РїРµСЂРµРґ РїСЂРѕСЂС–РґР¶СѓРІР°РЅРЅСЏРј")
    initial_parameters = count_parameters(model)
    logger.info(f"РљС–Р»СЊРєС–СЃС‚СЊ РїР°СЂР°РјРµС‚СЂС–РІ: {initial_parameters:,}")

    initial_accuracy = evaluate_model(model, val_loader, device)
    initial_inference_time = measure_inference_time(model, val_loader, device)

    # РџСЂРѕСЂС–РґР¶СѓРІР°РЅРЅСЏ РјРѕРґРµР»С–
    if args.pruning_method == "global":
        pruned_model = apply_global_pruning(model, args.amount)
    else:
        pruned_model = apply_local_pruning(model, args.amount)

    # РћС†С–РЅРєР° РјРѕРґРµР»С– РїС–СЃР»СЏ РїСЂРѕСЂС–РґР¶СѓРІР°РЅРЅСЏ
    logger.info("РћС†С–РЅРєР° РјРѕРґРµР»С– РїС–СЃР»СЏ РїСЂРѕСЂС–РґР¶СѓРІР°РЅРЅСЏ")
    pruned_accuracy = evaluate_model(pruned_model, val_loader, device)
    pruned_inference_time = measure_inference_time(pruned_model, val_loader, device)

    # Р”РѕРЅР°РІС‡Р°РЅРЅСЏ РјРѕРґРµР»С–, СЏРєС‰Рѕ РїРѕС‚СЂС–Р±РЅРѕ
    if args.fine_tune:
        pruned_model = fine_tune_pruned_model(
            pruned_model, train_loader, val_loader,
            num_epochs=args.fine_tune_epochs,
            learning_rate=args.fine_tune_lr
        )

        # РћС†С–РЅРєР° РјРѕРґРµР»С– РїС–СЃР»СЏ РґРѕРЅР°РІС‡Р°РЅРЅСЏ
        logger.info("РћС†С–РЅРєР° РјРѕРґРµР»С– РїС–СЃР»СЏ РґРѕРЅР°РІС‡Р°РЅРЅСЏ")
        fine_tuned_accuracy = evaluate_model(pruned_model, val_loader, device)
        fine_tuned_inference_time = measure_inference_time(pruned_model, val_loader, device)

    # Р’РёРґР°Р»РµРЅРЅСЏ РјР°СЃРѕРє РїСЂРѕСЂС–РґР¶СѓРІР°РЅРЅСЏ
    pruned_model = remove_pruning_masks(pruned_model)

    # Р—Р±РµСЂРµР¶РµРЅРЅСЏ РјРѕРґРµР»С–
    torch.save(pruned_model, args.save_path)
    logger.info(f"РџСЂРѕСЂС–РґР¶РµРЅР° РјРѕРґРµР»СЊ Р·Р±РµСЂРµР¶РµРЅР° РІ {args.save_path}")

    # Р’РёРІРµРґРµРЅРЅСЏ СЂРµР·СѓР»СЊС‚Р°С‚С–РІ
    logger.info("\nР РµР·СѓР»СЊС‚Р°С‚Рё РїСЂРѕСЂС–РґР¶СѓРІР°РЅРЅСЏ:")
    logger.info(f"РџРѕС‡Р°С‚РєРѕРІР° РєС–Р»СЊРєС–СЃС‚СЊ РїР°СЂР°РјРµС‚СЂС–РІ: {initial_parameters:,}")
    logger.info(f"РџРѕС‡Р°С‚РєРѕРІР° С‚РѕС‡РЅС–СЃС‚СЊ: {initial_accuracy:.2f}%")
    logger.info(f"РџРѕС‡Р°С‚РєРѕРІРёР№ С‡Р°СЃ С–РЅС„РµСЂРµРЅСЃСѓ: {initial_inference_time*1000:.2f} РјСЃ")

    final_parameters = count_parameters(pruned_model)
    logger.info(f"\nР¤С–РЅР°Р»СЊРЅР° РєС–Р»СЊРєС–СЃС‚СЊ РїР°СЂР°РјРµС‚СЂС–РІ: {final_parameters:,} ({final_parameters/initial_parameters*100:.2f}% РІС–Рґ РїРѕС‡Р°С‚РєРѕРІРѕС—)")
    logger.info(f"РўРѕС‡РЅС–СЃС‚СЊ РїС–СЃР»СЏ РїСЂРѕСЂС–РґР¶СѓРІР°РЅРЅСЏ: {pruned_accuracy:.2f}% ({pruned_accuracy-initial_accuracy:+.2f}%)")
    logger.info(f"Р§Р°СЃ С–РЅС„РµСЂРµРЅСЃСѓ РїС–СЃР»СЏ РїСЂРѕСЂС–РґР¶СѓРІР°РЅРЅСЏ: {pruned_inference_time*1000:.2f} РјСЃ ({(1-pruned_inference_time/initial_inference_time)*100:.2f}% РїСЂРёС€РІРёРґС€РµРЅРЅСЏ)")

    if args.fine_tune:
        logger.info(f"\nРўРѕС‡РЅС–СЃС‚СЊ РїС–СЃР»СЏ РґРѕРЅР°РІС‡Р°РЅРЅСЏ: {fine_tuned_accuracy:.2f}% ({fine_tuned_accuracy-initial_accuracy:+.2f}%)")
        logger.info(f"Р§Р°СЃ С–РЅС„РµСЂРµРЅСЃСѓ РїС–СЃР»СЏ РґРѕРЅР°РІС‡Р°РЅРЅСЏ: {fine_tuned_inference_time*1000:.2f} РјСЃ ({(1-fine_tuned_inference_time/initial_inference_time)*100:.2f}% РїСЂРёС€РІРёРґС€РµРЅРЅСЏ)")

if __name__ == "__main__":
    main()

