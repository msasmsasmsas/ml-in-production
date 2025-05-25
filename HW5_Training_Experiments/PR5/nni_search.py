#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
from tqdm import tqdm
import nni
import logging
import json
import argparse

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("nni_search")

# Импортируем класс датасета из PR1
import sys
sys.path.append('../PR1')
from train import AgriculturalRiskDataset

# Стандартная конфигурация
DEFAULT_CONFIG = {
    "data_path": "../crawler/downloads/",
    "image_dir": "../crawler/downloads/images/",
    "model_name": "resnet18",
    "num_epochs": 10,
    "batch_size": 32,
    "learning_rate": 0.001,
    "weight_decay": 1e-5,
    "seed": 42,
    "validation_split": 0.2,
    "risk_type": "diseases",  # "diseases", "pests", "weeds"
}

def get_default_params():
    """Получение параметров по умолчанию для NNI"""
    return {
        "model_name": "resnet18",
        "batch_size": 32,
        "learning_rate": 0.001,
        "weight_decay": 1e-5,
        "risk_type": "diseases"
    }

def get_model(model_name, num_classes):
    """
    Создание модели с предварительно обученными весами
    """
    if model_name == "resnet18":
        model = models.resnet18(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "resnet50":
        model = models.resnet50(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "mobilenet_v2":
        model = models.mobilenet_v2(pretrained=True)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
        
    return model

def create_data_loaders(config):
    """
    Создание загрузчиков данных для обучения и валидации
    """
    # Определяем трансформации для изображений
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Создаем полный датасет
    full_dataset = AgriculturalRiskDataset(
        csv_file=config["data_path"],
        image_dir=config["image_dir"],
        transform=None,  # Трансформация будет применена позже
        risk_type=config["risk_type"]
    )
    
    # Разделение на тренировочную и валидационную выборки
    train_indices, val_indices = train_test_split(
        range(len(full_dataset)),
        test_size=config["validation_split"],
        random_state=config["seed"],
        stratify=[full_dataset.data[i]["class_idx"] for i in range(len(full_dataset))]
    )
    
    # Создаем отдельные датасеты с соответствующими трансформациями
    train_dataset = torch.utils.data.Subset(
        AgriculturalRiskDataset(
            csv_file=config["data_path"],
            image_dir=config["image_dir"],
            transform=train_transform,
            risk_type=config["risk_type"]
        ),
        train_indices
    )
    
    val_dataset = torch.utils.data.Subset(
        AgriculturalRiskDataset(
            csv_file=config["data_path"],
            image_dir=config["image_dir"],
            transform=val_transform,
            risk_type=config["risk_type"]
        ),
        val_indices
    )
    
    # Создаем загрузчики данных
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    class_names = full_dataset.classes
    num_classes = len(class_names)
    
    return train_loader, val_loader, num_classes, class_names

def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """
    Обучение модели на одну эпоху
    """
    model.train()
    running_loss = 0.0
    correct_preds = 0
    total_samples = 0
    
    progress_bar = tqdm(train_loader, desc="Training", disable=True)
    
    for inputs, labels in progress_bar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Обнуляем градиенты
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass и оптимизация
        loss.backward()
        optimizer.step()
        
        # Статистика
        running_loss += loss.item() * inputs.size(0)
        
        _, predicted = torch.max(outputs, 1)
        correct_preds += (predicted == labels).sum().item()
        total_samples += labels.size(0)
    
    epoch_loss = running_loss / total_samples
    epoch_acc = correct_preds / total_samples
    
    return epoch_loss, epoch_acc

def validate(model, val_loader, criterion, device):
    """
    Валидация модели
    """
    model.eval()
    running_loss = 0.0
    correct_preds = 0
    total_samples = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Validation", disable=True):
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Статистика
            running_loss += loss.item() * inputs.size(0)
            
            _, predicted = torch.max(outputs, 1)
            correct_preds += (predicted == labels).sum().item()
            total_samples += labels.size(0)
            
            # Сохраняем для метрик
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    val_loss = running_loss / total_samples
    val_acc = correct_preds / total_samples
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    return val_loss, val_acc, f1, all_preds, all_labels

def train_model(params):
    """
    Обучение модели с параметрами из NNI
    """
    # Объединяем параметры NNI с параметрами по умолчанию
    config = DEFAULT_CONFIG.copy()
    config.update(params)
    
    # Логируем параметры
    logger.info(f"Training with parameters: {config}")
    
    # Настройка seed для воспроизводимости
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    
    # Определяем устройство
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Создаем загрузчики данных
    train_loader, val_loader, num_classes, class_names = create_data_loaders(config)
    
    logger.info(f"Number of classes: {num_classes}")
    logger.info(f"Training samples: {len(train_loader.dataset)}")
    logger.info(f"Validation samples: {len(val_loader.dataset)}")
    
    # Создаем модель
    model = get_model(config["model_name"], num_classes)
    model = model.to(device)
    
    # Определяем функцию потерь и оптимизатор
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"]
    )
    
    # Создаем папку для сохранения моделей
    os.makedirs("models", exist_ok=True)
    
    # Обучение модели
    best_val_f1 = 0.0
    early_stop_count = 0
    
    for epoch in range(config["num_epochs"]):
        logger.info(f"Epoch {epoch+1}/{config['num_epochs']}")
        
        # Обучение
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        
        # Валидация
        val_loss, val_acc, val_f1, all_preds, all_labels = validate(model, val_loader, criterion, device)
        
        logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")
        
        # Отправляем метрики в NNI
        nni.report_intermediate_result({
            "default": val_f1,
            "val_f1": val_f1,
            "val_acc": val_acc,
            "val_loss": val_loss,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "epoch": epoch + 1
        })
        
        # Сохраняем лучшую модель
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), f"models/{config['risk_type']}_{config['model_name']}_nni_best.pt")
            early_stop_count = 0
        else:
            early_stop_count += 1
        
        # Ранняя остановка если 3 эпохи нет улучшения
        if early_stop_count >= 3:
            logger.info(f"Early stopping after {epoch+1} epochs")
            break
    
    # Сохраняем метаданные модели
    metadata = {
        "model_name": config["model_name"],
        "num_classes": num_classes,
        "class_names": class_names.tolist(),
        "risk_type": config["risk_type"],
        "best_val_f1": best_val_f1,
        "params": config
    }
    
    with open(f"models/{config['risk_type']}_{config['model_name']}_nni_metadata.json", "w") as f:
        json.dump(metadata, f)
    
    # Отправляем итоговый результат в NNI
    nni.report_final_result({
        "default": best_val_f1,
        "best_val_f1": best_val_f1
    })
    
    return best_val_f1

def generate_config_file():
    """Генерация конфигурационного файла для NNI"""
    config = {
        "search_space": {
            "model_name": {"_type": "choice", "_value": ["resnet18", "resnet50", "mobilenet_v2"]},
            "batch_size": {"_type": "choice", "_value": [16, 32, 64]},
            "learning_rate": {"_type": "loguniform", "_value": [1e-4, 1e-2]},
            "weight_decay": {"_type": "loguniform", "_value": [1e-6, 1e-3]},
            "risk_type": {"_type": "choice", "_value": ["diseases", "pests", "weeds"]}
        },
        "trial_command": "python3 nni_search.py --trial",
        "trial_code_directory": ".",
        "trial_concurrency": 2,
        "experiment_working_directory": "./nni_experiments",
        "training_service": {
            "platform": "local"
        },
        "experiment_name": "agri_risk_classification",
        "max_trial_number": 20,
        "max_experiment_duration": "12h",
        "tuner": {
            "name": "TPE",
            "class_args": {
                "optimize_mode": "maximize"
            }
        },
        "assessor": {
            "name": "Medianstop",
            "class_args": {
                "optimize_mode": "maximize"
            }
        }
    }
    
    with open("nni_config.json", "w") as f:
        json.dump(config, f, indent=4)
    
    print("NNI config generated: nni_config.json")
    print("To start the experiment, run: nnictl create --config nni_config.json")
    
    return "nni_config.json"

def main():
    parser = argparse.ArgumentParser(description="Run NNI hyperparameter search")
    parser.add_argument("--trial", action="store_true", help="Run as NNI trial")
    parser.add_argument("--generate_config", action="store_true", help="Generate NNI config file")
    parser.add_argument("--manual", action="store_true", help="Run with default parameters (no NNI)")
    
    args = parser.parse_args()
    
    if args.generate_config:
        generate_config_file()
        return
    
    if args.trial:
        # Получаем параметры от NNI
        params = nni.get_next_parameter()
        train_model(params)
    elif args.manual:
        # Запуск с параметрами по умолчанию
        train_model(get_default_params())
    else:
        print("Please specify one of the options: --trial, --generate_config, or --manual")
        parser.print_help()

if __name__ == "__main__":
    main()
