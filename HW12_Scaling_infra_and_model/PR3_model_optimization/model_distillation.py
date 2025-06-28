# новлена версія для PR
#!/usr/bin/env python

'''
Р”РёСЃС‚РёР»СЏС†С–СЏ Р·РЅР°РЅСЊ Р· РІРµР»РёРєРѕС— РјРѕРґРµР»С– РґРѕ РјР°Р»РѕС—
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
from torch.utils.data import DataLoader, Dataset

import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights, resnet18, ResNet18_Weights

# РќР°Р»Р°С€С‚СѓРІР°РЅРЅСЏ Р»РѕРіСѓРІР°РЅРЅСЏ
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('model_distillation')

class DistillationLoss(nn.Module):
    '''
    Р¤СѓРЅРєС†С–СЏ РІС‚СЂР°С‚Рё РґР»СЏ РґРёСЃС‚РёР»СЏС†С–С— Р·РЅР°РЅСЊ
    РљРѕРјР±С–РЅСѓС” РєСЂРѕСЃ-РµРЅС‚СЂРѕРїС–СЋ С‚Р° KL-РґРёРІРµСЂРіРµРЅС†С–СЋ
    '''
    def __init__(self, alpha=0.5, temperature=2.0):
        super(DistillationLoss, self).__init__()
        self.alpha = alpha  # Р’Р°РіР° РјС–Р¶ Рј'СЏРєРёРјРё С‚Р° Р¶РѕСЂСЃС‚РєРёРјРё РјС–С‚РєР°РјРё
        self.temperature = temperature  # РўРµРјРїРµСЂР°С‚СѓСЂР° РґР»СЏ Рј'СЏРєРёС… РјС–С‚РѕРє

    def forward(self, student_outputs, teacher_outputs, labels):
        '''
        РћР±С‡РёСЃР»РµРЅРЅСЏ РІС‚СЂР°С‚Рё РґРёСЃС‚РёР»СЏС†С–С—

        РџР°СЂР°РјРµС‚СЂРё:
        -----------
        student_outputs: РІРёС…РѕРґРё РјРѕРґРµР»С– СЃС‚СѓРґРµРЅС‚Р°
        teacher_outputs: РІРёС…РѕРґРё РјРѕРґРµР»С– РІС‡РёС‚РµР»СЏ
        labels: СЃРїСЂР°РІР¶РЅС– РјС–С‚РєРё

        РџРѕРІРµСЂС‚Р°С”:
        -----------
        РєРѕРјР±С–РЅРѕРІР°РЅСѓ РІС‚СЂР°С‚Сѓ
        '''
        # Р–РѕСЂСЃС‚РєС– РјС–С‚РєРё: РєСЂРѕСЃ-РµРЅС‚СЂРѕРїС–СЏ Р· СЃРїСЂР°РІР¶РЅС–РјРё РјС–С‚РєР°РјРё
        hard_loss = F.cross_entropy(student_outputs, labels)

        # Рњ'СЏРєС– РјС–С‚РєРё: KL-РґРёРІРµСЂРіРµРЅС†С–СЏ РјС–Р¶ РІРёС…РѕРґР°РјРё СЃС‚СѓРґРµРЅС‚Р° С‚Р° РІС‡РёС‚РµР»СЏ
        soft_student = F.log_softmax(student_outputs / self.temperature, dim=1)
        soft_teacher = F.softmax(teacher_outputs / self.temperature, dim=1)
        soft_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean') * (self.temperature ** 2)

        # РљРѕРјР±С–РЅРѕРІР°РЅР° РІС‚СЂР°С‚Р°
        loss = (1 - self.alpha) * hard_loss + self.alpha * soft_loss

        return loss

def load_teacher_model(model_path=None):
    '''
    Р—Р°РІР°РЅС‚Р°Р¶РµРЅРЅСЏ РјРѕРґРµР»С– РІС‡РёС‚РµР»СЏ (РІРµР»РёРєРѕС— РјРѕРґРµР»С–)

    РџР°СЂР°РјРµС‚СЂРё:
    -----------
    model_path: С€Р»СЏС… РґРѕ Р·Р±РµСЂРµР¶РµРЅРѕС— РјРѕРґРµР»С– (СЏРєС‰Рѕ None, РІРёРєРѕСЂРёСЃС‚РѕРІСѓС”С‚СЊСЃСЏ РїРѕРїРµСЂРµРґРЅСЊРѕ РЅР°РІС‡РµРЅР° РјРѕРґРµР»СЊ)

    РџРѕРІРµСЂС‚Р°С”:
    -----------
    РјРѕРґРµР»СЊ РІС‡РёС‚РµР»СЏ
    '''
    if model_path and os.path.exists(model_path):
        logger.info(f"Р—Р°РІР°РЅС‚Р°Р¶РµРЅРЅСЏ РјРѕРґРµР»С– РІС‡РёС‚РµР»СЏ Р· {model_path}")
        teacher = torch.load(model_path)
    else:
        logger.info("Р—Р°РІР°РЅС‚Р°Р¶РµРЅРЅСЏ РїРѕРїРµСЂРµРґРЅСЊРѕ РЅР°РІС‡РµРЅРѕС— РјРѕРґРµР»С– ResNet50")
        teacher = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

    # РџРµСЂРµРІРµРґРµРЅРЅСЏ РјРѕРґРµР»С– РІ СЂРµР¶РёРј РѕС†С–РЅРєРё
    teacher.eval()

    return teacher

def create_student_model(num_classes=1000):
    '''
    РЎС‚РІРѕСЂРµРЅРЅСЏ РјРѕРґРµР»С– СЃС‚СѓРґРµРЅС‚Р° (РјР°Р»РѕС— РјРѕРґРµР»С–)

    РџР°СЂР°РјРµС‚СЂРё:
    -----------
    num_classes: РєС–Р»СЊРєС–СЃС‚СЊ РєР»Р°СЃС–РІ РґР»СЏ РєР»Р°СЃРёС„С–РєР°С†С–С—

    РџРѕРІРµСЂС‚Р°С”:
    -----------
    РјРѕРґРµР»СЊ СЃС‚СѓРґРµРЅС‚Р°
    '''
    logger.info("РЎС‚РІРѕСЂРµРЅРЅСЏ РјРѕРґРµР»С– СЃС‚СѓРґРµРЅС‚Р° ResNet18")
    student = resnet18(weights=None)

    # РђРґР°РїС‚Р°С†С–СЏ РѕСЃС‚Р°РЅРЅСЊРѕРіРѕ С€Р°СЂСѓ РґР»СЏ РєС–Р»СЊРєРѕСЃС‚С– РєР»Р°СЃС–РІ
    if student.fc.out_features != num_classes:
        in_features = student.fc.in_features
        student.fc = nn.Linear(in_features, num_classes)

    return student

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
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
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

def distill_model(teacher, student, train_loader, val_loader, 
                num_epochs=10, learning_rate=0.01, temperature=2.0, alpha=0.5,
                save_path='distilled_model.pt'):
    '''
    Р”РёСЃС‚РёР»СЏС†С–СЏ Р·РЅР°РЅСЊ Р· РјРѕРґРµР»С– РІС‡РёС‚РµР»СЏ РґРѕ РјРѕРґРµР»С– СЃС‚СѓРґРµРЅС‚Р°

    РџР°СЂР°РјРµС‚СЂРё:
    -----------
    teacher: РјРѕРґРµР»СЊ РІС‡РёС‚РµР»СЏ
    student: РјРѕРґРµР»СЊ СЃС‚СѓРґРµРЅС‚Р°
    train_loader: Р·Р°РІР°РЅС‚Р°Р¶СѓРІР°С‡ РґР°РЅРёС… РґР»СЏ РЅР°РІС‡Р°РЅРЅСЏ
    val_loader: Р·Р°РІР°РЅС‚Р°Р¶СѓРІР°С‡ РґР°РЅРёС… РґР»СЏ РІР°Р»С–РґР°С†С–С—
    num_epochs: РєС–Р»СЊРєС–СЃС‚СЊ РµРїРѕС… РЅР°РІС‡Р°РЅРЅСЏ
    learning_rate: С€РІРёРґРєС–СЃС‚СЊ РЅР°РІС‡Р°РЅРЅСЏ
    temperature: С‚РµРјРїРµСЂР°С‚СѓСЂР° РґР»СЏ Рј'СЏРєРёС… РјС–С‚РѕРє
    alpha: РІР°РіР° РјС–Р¶ Рј'СЏРєРёРјРё С‚Р° Р¶РѕСЂСЃС‚РєРёРјРё РјС–С‚РєР°РјРё
    save_path: С€Р»СЏС… РґР»СЏ Р·Р±РµСЂРµР¶РµРЅРЅСЏ РјРѕРґРµР»С–

    РџРѕРІРµСЂС‚Р°С”:
    -----------
    РЅР°РІС‡РµРЅР° РјРѕРґРµР»СЊ СЃС‚СѓРґРµРЅС‚Р°
    '''
    # Р’РёР·РЅР°С‡РµРЅРЅСЏ РїСЂРёСЃС‚СЂРѕСЋ РґР»СЏ РЅР°РІС‡Р°РЅРЅСЏ
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Р’РёРєРѕСЂРёСЃС‚Р°РЅРЅСЏ РїСЂРёСЃС‚СЂРѕСЋ: {device}")

    # РџРµСЂРµРјС–С‰РµРЅРЅСЏ РјРѕРґРµР»РµР№ РЅР° РїСЂРёСЃС‚СЂС–Р№
    teacher = teacher.to(device)
    student = student.to(device)

    # Р—Р°РјРѕСЂРѕР¶СѓРІР°РЅРЅСЏ РїР°СЂР°РјРµС‚СЂС–РІ РјРѕРґРµР»С– РІС‡РёС‚РµР»СЏ
    for param in teacher.parameters():
        param.requires_grad = False

    # Р’РёР·РЅР°С‡РµРЅРЅСЏ РѕРїС‚РёРјС–Р·Р°С‚РѕСЂР°
    optimizer = optim.SGD(student.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # Р’РёР·РЅР°С‡РµРЅРЅСЏ С„СѓРЅРєС†С–С— РІС‚СЂР°С‚Рё
    criterion = DistillationLoss(alpha=alpha, temperature=temperature)

    # РќР°РІС‡Р°РЅРЅСЏ РјРѕРґРµР»С–
    best_acc = 0.0
    for epoch in range(num_epochs):
        logger.info(f"Р•РїРѕС…Р° {epoch+1}/{num_epochs}")

        # РќР°РІС‡Р°Р»СЊРЅР° С„Р°Р·Р°
        student.train()
        train_loss = 0.0
        train_acc = 0.0

        train_bar = tqdm(train_loader, desc=f"Р•РїРѕС…Р° {epoch+1}/{num_epochs} [РќР°РІС‡Р°РЅРЅСЏ]")
        for inputs, labels in train_bar:
            inputs, labels = inputs.to(device), labels.to(device)

            # РћС‡РёС‰РµРЅРЅСЏ РіСЂР°РґС–С”РЅС‚С–РІ
            optimizer.zero_grad()

            # РћС‚СЂРёРјР°РЅРЅСЏ РІРёС…РѕРґС–РІ РІС–Рґ РјРѕРґРµР»РµР№
            with torch.no_grad():
                teacher_outputs = teacher(inputs)

            student_outputs = student(inputs)

            # РћР±С‡РёСЃР»РµРЅРЅСЏ РІС‚СЂР°С‚Рё
            loss = criterion(student_outputs, teacher_outputs, labels)

            # РћР±С‡РёСЃР»РµРЅРЅСЏ РіСЂР°РґС–С”РЅС‚С–РІ С– РѕРЅРѕРІР»РµРЅРЅСЏ РїР°СЂР°РјРµС‚СЂС–РІ
            loss.backward()
            optimizer.step()

            # РЎС‚Р°С‚РёСЃС‚РёРєР°
            train_loss += loss.item() * inputs.size(0)

            # РћР±С‡РёСЃР»РµРЅРЅСЏ С‚РѕС‡РЅРѕСЃС‚С–
            _, preds = torch.max(student_outputs, 1)
            train_acc += torch.sum(preds == labels.data)

            # РћРЅРѕРІР»РµРЅРЅСЏ РїСЂРѕРіСЂРµСЃ-Р±Р°СЂСѓ
            train_bar.set_postfix({'loss': loss.item()})

        # РћРЅРѕРІР»РµРЅРЅСЏ РїР»Р°РЅСѓРІР°Р»СЊРЅРёРєР° С€РІРёРґРєРѕСЃС‚С– РЅР°РІС‡Р°РЅРЅСЏ
        scheduler.step()

        # РЎС‚Р°С‚РёСЃС‚РёРєР° Р·Р° РµРїРѕС…Сѓ
        train_loss = train_loss / len(train_loader.dataset)
        train_acc = train_acc.float() / len(train_loader.dataset)

        # Р’Р°Р»С–РґР°С†С–Р№РЅР° С„Р°Р·Р°
        student.eval()
        val_loss = 0.0
        val_acc = 0.0

        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f"Р•РїРѕС…Р° {epoch+1}/{num_epochs} [Р’Р°Р»С–РґР°С†С–СЏ]")
            for inputs, labels in val_bar:
                inputs, labels = inputs.to(device), labels.to(device)

                # РћС‚СЂРёРјР°РЅРЅСЏ РІРёС…РѕРґС–РІ РІС–Рґ РјРѕРґРµР»РµР№
                teacher_outputs = teacher(inputs)
                student_outputs = student(inputs)

                # РћР±С‡РёСЃР»РµРЅРЅСЏ РІС‚СЂР°С‚Рё
                loss = criterion(student_outputs, teacher_outputs, labels)

                # РЎС‚Р°С‚РёСЃС‚РёРєР°
                val_loss += loss.item() * inputs.size(0)

                # РћР±С‡РёСЃР»РµРЅРЅСЏ С‚РѕС‡РЅРѕСЃС‚С–
                _, preds = torch.max(student_outputs, 1)
                val_acc += torch.sum(preds == labels.data)

                # РћРЅРѕРІР»РµРЅРЅСЏ РїСЂРѕРіСЂРµСЃ-Р±Р°СЂСѓ
                val_bar.set_postfix({'loss': loss.item()})

        # РЎС‚Р°С‚РёСЃС‚РёРєР° Р·Р° РµРїРѕС…Сѓ
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_acc.float() / len(val_loader.dataset)

        logger.info(f"Р•РїРѕС…Р° {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Р—Р±РµСЂРµР¶РµРЅРЅСЏ РЅР°Р№РєСЂР°С‰РѕС— РјРѕРґРµР»С–
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(student, save_path)
            logger.info(f"Р—Р±РµСЂРµР¶РµРЅРѕ РєСЂР°С‰Сѓ РјРѕРґРµР»СЊ Р· С‚РѕС‡РЅС–СЃС‚СЋ РІР°Р»С–РґР°С†С–С—: {val_acc:.4f}")

    logger.info(f"Р”РёСЃС‚РёР»СЏС†С–СЏ Р·Р°РІРµСЂС€РµРЅР°. РќР°Р№РєСЂР°С‰Р° С‚РѕС‡РЅС–СЃС‚СЊ РІР°Р»С–РґР°С†С–С—: {best_acc:.4f}")

    # Р—Р°РІР°РЅС‚Р°Р¶РµРЅРЅСЏ РЅР°Р№РєСЂР°С‰РѕС— РјРѕРґРµР»С–
    best_student = torch.load(save_path)

    return best_student

def main():
    parser = argparse.ArgumentParser(description="Р”РёСЃС‚РёР»СЏС†С–СЏ Р·РЅР°РЅСЊ Р· РІРµР»РёРєРѕС— РјРѕРґРµР»С– РґРѕ РјР°Р»РѕС—")
    parser.add_argument("--teacher", type=str, default=None,
                        help="РЁР»СЏС… РґРѕ РјРѕРґРµР»С– РІС‡РёС‚РµР»СЏ (СЏРєС‰Рѕ None, РІРёРєРѕСЂРёСЃС‚РѕРІСѓС”С‚СЊСЃСЏ РїРѕРїРµСЂРµРґРЅСЊРѕ РЅР°РІС‡РµРЅР° РјРѕРґРµР»СЊ)")
    parser.add_argument("--data-dir", type=str, default="./data",
                        help="Р”РёСЂРµРєС‚РѕСЂС–СЏ Р· РґР°РЅРёРјРё")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Р РѕР·РјС–СЂ Р±Р°С‚С‡Сѓ")
    parser.add_argument("--epochs", type=int, default=10,
                        help="РљС–Р»СЊРєС–СЃС‚СЊ РµРїРѕС… РЅР°РІС‡Р°РЅРЅСЏ")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="РЁРІРёРґРєС–СЃС‚СЊ РЅР°РІС‡Р°РЅРЅСЏ")
    parser.add_argument("--temperature", type=float, default=2.0,
                        help="РўРµРјРїРµСЂР°С‚СѓСЂР° РґР»СЏ Рј'СЏРєРёС… РјС–С‚РѕРє")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Р’Р°РіР° РјС–Р¶ Рј'СЏРєРёРјРё С‚Р° Р¶РѕСЂСЃС‚РєРёРјРё РјС–С‚РєР°РјРё")
    parser.add_argument("--save-path", type=str, default="distilled_model.pt",
                        help="РЁР»СЏС… РґР»СЏ Р·Р±РµСЂРµР¶РµРЅРЅСЏ РјРѕРґРµР»С–")

    args = parser.parse_args()

    # Р—Р°РІР°РЅС‚Р°Р¶РµРЅРЅСЏ РјРѕРґРµР»С– РІС‡РёС‚РµР»СЏ
    teacher = load_teacher_model(args.teacher)

    # РЎС‚РІРѕСЂРµРЅРЅСЏ РјРѕРґРµР»С– СЃС‚СѓРґРµРЅС‚Р°
    student = create_student_model()

    # РћС‚СЂРёРјР°РЅРЅСЏ Р·Р°РІР°РЅС‚Р°Р¶СѓРІР°С‡С–РІ РґР°РЅРёС…
    train_loader, val_loader = get_dataloaders(args.data_dir, args.batch_size)

    # Р”РёСЃС‚РёР»СЏС†С–СЏ РјРѕРґРµР»С–
    distilled_student = distill_model(
        teacher, student, train_loader, val_loader,
        num_epochs=args.epochs, learning_rate=args.lr,
        temperature=args.temperature, alpha=args.alpha,
        save_path=args.save_path
    )

    logger.info(f"РњРѕРґРµР»СЊ СѓСЃРїС–С€РЅРѕ РґРёСЃС‚РёР»СЊРѕРІР°РЅР° С‚Р° Р·Р±РµСЂРµР¶РµРЅР° РІ {args.save_path}")

if __name__ == "__main__":
    main()

