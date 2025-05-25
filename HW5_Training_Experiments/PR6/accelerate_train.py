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
import wandb
import argparse
import json
from accelerate import Accelerator, DistributedType
from accelerate.utils import set_seed

# Импортируем класс датасета из PR1
import sys
sys.path.append('../PR1')
from train import AgriculturalRiskDataset

# Настройка логирования
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("accelerate_train")

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
    "wandb_project": "agri_risk_classification_accelerate",
    "wandb_entity": None,
    "checkpoint_dir": "./models",
    "gradient_accumulation_steps": 1,
    "mixed_precision": "fp16"  # "no", "fp16", "bf16"
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

def train_and_evaluate(config):
    """
    Основная функция для обучения и оценки модели с Accelerate
    """
    # Инициализируем Accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        mixed_precision=config["mixed_precision"]
    )
    
    # Показываем информацию о распределенном окружении
    if accelerator.is_main_process:
        logger.info(f"Distributed environment: {accelerator.distributed_type}")
        logger.info(f"Mixed precision: {accelerator.mixed_precision}")
        logger.info(f"Number of processes: {accelerator.num_processes}")
        
        # Инициализируем WandB для отслеживания экспериментов (только на главном процессе)
        if config.get("use_wandb", False):
            wandb.init(
                project=config["wandb_project"],
                entity=config["wandb_entity"],
                config=config,
                name=f"{config['risk_type']}_{config['model_name']}_accelerate"
            )
    
    # Устанавливаем seed для воспроизводимости
    set_seed(config["seed"])
    
    # Создаем загрузчики данных
    train_loader, val_loader, num_classes, class_names = create_data_loaders(config)
    
    if accelerator.is_main_process:
        logger.info(f"Number of classes: {num_classes}")
        logger.info(f"Training samples: {len(train_loader.dataset)}")
        logger.info(f"Validation samples: {len(val_loader.dataset)}")
    
    # Создаем модель, функцию потерь и оптимизатор
    model = get_model(config["model_name"], num_classes)
    
    # Включаем checkpoint gradients для экономии памяти (опционально)
    if config.get("gradient_checkpointing", False):
        model.gradient_checkpointing_enable()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"]
    )
    
    # Подготавливаем все для Accelerate
    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )
    
    # Создаем папку для сохранения моделей
    if accelerator.is_main_process:
        os.makedirs(config["checkpoint_dir"], exist_ok=True)
    
    # Обучение модели
    best_val_f1 = 0.0
    
    for epoch in range(config["num_epochs"]):
        if accelerator.is_main_process:
            logger.info(f"Epoch {epoch+1}/{config['num_epochs']}")
        
        # --- Обучение ---
        model.train()
        train_loss = 0.0
        correct_preds = 0
        total_samples = 0
        
        progress_bar = tqdm(train_loader, disable=not accelerator.is_main_process)
        
        for step, (inputs, labels) in enumerate(progress_bar):
            # Аккумулируем градиенты вручную только если не используем Accelerator
            # с настроенными gradient_accumulation_steps
            with accelerator.accumulate(model):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
            
            # Собираем статистику
            train_loss += loss.item() * inputs.size(0)
            
            _, predicted = torch.max(outputs, 1)
            correct_batch = (predicted == labels).sum().item()
            correct_preds += correct_batch
            total_samples += labels.size(0)
            
            if accelerator.is_main_process:
                progress_bar.set_description(f"Epoch {epoch+1}")
                progress_bar.set_postfix({
                    "loss": loss.item(),
                    "acc": correct_batch / labels.size(0)
                })
        
        train_loss = train_loss / total_samples
        train_acc = correct_preds / total_samples
        
        # --- Валидация ---
        model.eval()
        val_loss = 0.0
        correct_preds = 0
        total_samples = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, disable=not accelerator.is_main_process):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                # Статистика
                val_loss += loss.item() * inputs.size(0)
                
                _, predicted = torch.max(outputs, 1)
                correct_preds += (predicted == labels).sum().item()
                total_samples += labels.size(0)
                
                # Собираем предсказания и метки для F1
                all_preds.append(accelerator.gather(predicted))
                all_labels.append(accelerator.gather(labels))
        
        # Объединяем предсказания и метки со всех процессов
        all_preds = torch.cat(all_preds).cpu().numpy()
        all_labels = torch.cat(all_labels).cpu().numpy()
        
        # Вычисляем метрики
        val_loss = val_loss / total_samples
        val_acc = correct_preds / total_samples
        val_f1 = f1_score(all_labels[:len(val_loader.dataset)], all_preds[:len(val_loader.dataset)], average='weighted')
        
        if accelerator.is_main_process:
            logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")
            
            # Отправляем метрики в WandB
            if config.get("use_wandb", False):
                wandb.log({
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "val_f1": val_f1,
                    "epoch": epoch + 1
                })
            
            # Сохраняем лучшую модель
            is_best = val_f1 > best_val_f1
            if is_best:
                best_val_f1 = val_f1
                
                # Сохраняем модель
                unwrapped_model = accelerator.unwrap_model(model)
                checkpoint_path = os.path.join(
                    config["checkpoint_dir"],
                    f"{config['risk_type']}_{config['model_name']}_accelerate_best.pt"
                )
                
                accelerator.save({
                    "model_state_dict": unwrapped_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "val_f1": val_f1,
                    "config": config,
                    "class_names": class_names
                }, checkpoint_path)
                
                logger.info(f"Saved best model with F1: {val_f1:.4f}")
    
    # Завершаем эксперимент
    if accelerator.is_main_process and config.get("use_wandb", False):
        wandb.finish()
    
    return best_val_f1

def main():
    parser = argparse.ArgumentParser(description="Train with Hugging Face Accelerate")
    parser.add_argument("--model", type=str, default="resnet18", choices=["resnet18", "resnet50", "mobilenet_v2"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--risk_type", type=str, default="diseases", choices=["diseases", "pests", "weeds"])
    parser.add_argument("--use_wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Enable gradient checkpointing")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--config_file", type=str, help="Path to JSON config file")
    
    args = parser.parse_args()
    
    # Загружаем конфигурацию из файла, если указан
    if args.config_file:
        with open(args.config_file, 'r') as f:
            config = json.load(f)
    else:
        # Создаем конфигурацию из аргументов
        config = DEFAULT_CONFIG.copy()
        config.update({
            "model_name": args.model,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "num_epochs": args.epochs,
            "risk_type": args.risk_type,
            "use_wandb": args.use_wandb,
            "gradient_checkpointing": args.gradient_checkpointing,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "mixed_precision": args.mixed_precision,
            "seed": args.seed
        })
    
    # Запускаем обучение
    train_and_evaluate(config)

if __name__ == "__main__":
    main()
