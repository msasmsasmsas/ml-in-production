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
import wandb
import argparse
import json
import tempfile
from typing import Dict

# Импортируем ray и ray train
import ray
from ray import train
from ray.train import Trainer, TrainingCallback
from ray.train.torch import TorchTrainer, TorchConfig

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
logger = logging.getLogger("ray_train")

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
    "wandb_project": "agri_risk_classification_ray",
    "wandb_entity": None,
    "checkpoint_dir": "./models",
    "use_gpu": True,
    "num_workers": 2
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

def prepare_data(config):
    """
    Подготовка данных для Ray Train
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
    
    class_names = full_dataset.classes
    num_classes = len(class_names)
    
    return train_dataset, val_dataset, num_classes, class_names

# Callback для отслеживания в WandB
class WandbCallback(TrainingCallback):
    def __init__(self, config):
        self.config = config
        self.initialized = False
    
    def setup(self, **kwargs):
        if not self.initialized and self.config.get("use_wandb", False):
            wandb.init(
                project=self.config["wandb_project"],
                entity=self.config["wandb_entity"],
                config=self.config,
                name=f"{self.config['risk_type']}_{self.config['model_name']}_ray"
            )
            self.initialized = True
    
    def handle_result(self, results, **kwargs):
        if self.initialized and self.config.get("use_wandb", False):
            wandb.log(results)
    
    def on_checkpoint(self, checkpoint, **kwargs):
        pass
    
    def on_dataset_uploaded(self, **kwargs):
        pass
    
    def teardown(self):
        if self.initialized and self.config.get("use_wandb", False):
            wandb.finish()

def train_func(config):
    """
    Функция обучения для Ray Train
    """
    # Настройка устройства
    device = torch.device("cuda" if torch.cuda.is_available() and config["use_gpu"] else "cpu")
    
    # Получаем информацию о распределенном окружении Ray Train
    rank = train.get_context().get_world_rank()
    world_size = train.get_context().get_world_size()
    
    logger.info(f"Training worker {rank+1}/{world_size} on {device}")
    
    # Загружаем данные из состояния Ray Train
    train_dataset = train.get_dataset_shard("train")
    val_dataset = train.get_dataset_shard("val")
    
    # Создаем загрузчики данных
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True if device.type == "cuda" else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=4,
        pin_memory=True if device.type == "cuda" else False
    )
    
    # Создаем модель
    model = get_model(config["model_name"], config["num_classes"])
    model = model.to(device)
    
    # Определяем функцию потерь и оптимизатор
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"]
    )
    
    # Обучение модели
    best_val_f1 = 0.0
    
    for epoch in range(config["num_epochs"]):
        # --- Обучение ---
        model.train()
        train_loss = 0.0
        correct_preds = 0
        total_samples = 0
        
        for inputs, labels in train_loader:
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
            train_loss += loss.item() * inputs.size(0)
            
            _, predicted = torch.max(outputs, 1)
            correct_preds += (predicted == labels).sum().item()
            total_samples += labels.size(0)
        
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
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                # Статистика
                val_loss += loss.item() * inputs.size(0)
                
                _, predicted = torch.max(outputs, 1)
                correct_preds += (predicted == labels).sum().item()
                total_samples += labels.size(0)
                
                # Сохраняем для метрик
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        val_loss = val_loss / total_samples
        val_acc = correct_preds / total_samples
        val_f1 = f1_score(all_labels, all_preds, average='weighted')
        
        # Логируем метрики
        metrics = {
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_f1": val_f1,
            "epoch": epoch + 1
        }
        
        # Отправляем метрики в Ray Train
        train.report(metrics)
        
        # Сохраняем лучшую модель
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            
            # Сохраняем чекпоинт модели
            with tempfile.TemporaryDirectory() as temp_dir:
                checkpoint_path = os.path.join(temp_dir, "model.pt")
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "val_f1": val_f1,
                    "config": config,
                    "class_names": config["class_names"]
                }, checkpoint_path)
                
                # Сохраняем чекпоинт через Ray Train
                train.save_checkpoint(checkpoint=checkpoint_path)
    
    return {"best_val_f1": best_val_f1}

def train_model_with_ray(config):
    """
    Запуск обучения модели с Ray Train
    """
    # Инициализируем Ray, если еще не инициализирован
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
    
    # Подготавливаем данные
    train_dataset, val_dataset, num_classes, class_names = prepare_data(config)
    
    # Добавляем информацию о классах в конфигурацию
    config["num_classes"] = num_classes
    config["class_names"] = class_names
    
    logger.info(f"Number of classes: {num_classes}")
    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(val_dataset)}")
    
    # Создаем тренера Ray Train
    trainer = TorchTrainer(
        train_loop_per_worker=train_func,
        train_loop_config=config,
        scaling_config={"num_workers": config["num_workers"], "use_gpu": config["use_gpu"]},
        datasets={
            "train": ray.data.from_torch(train_dataset),
            "val": ray.data.from_torch(val_dataset)
        },
        run_config=train.RunConfig(
            callbacks=[WandbCallback(config)],
            storage_path=config["checkpoint_dir"],
            name=f"{config['risk_type']}_{config['model_name']}_ray"
        )
    )
    
    # Запускаем обучение
    result = trainer.fit()
    
    # Получаем и сохраняем лучшую модель
    checkpoint = result.checkpoint
    if checkpoint:
        checkpoint_path = trainer.get_checkpoint().path
        logger.info(f"Best checkpoint saved at: {checkpoint_path}")
        
        # Копируем чекпоинт в целевую директорию
        target_path = os.path.join(
            config["checkpoint_dir"],
            f"{config['risk_type']}_{config['model_name']}_ray_best.pt"
        )
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        
        # Загружаем чекпоинт, чтобы получить содержимое
        ckpt_data = torch.load(checkpoint_path)
        torch.save(ckpt_data, target_path)
        
        logger.info(f"Saved final model to {target_path}")
    
    return result

def main():
    parser = argparse.ArgumentParser(description="Distributed training with Ray")
    parser.add_argument("--model", type=str, default="resnet18", choices=["resnet18", "resnet50", "mobilenet_v2"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--risk_type", type=str, default="diseases", choices=["diseases", "pests", "weeds"])
    parser.add_argument("--use_wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of Ray workers")
    parser.add_argument("--use_gpu", action="store_true", help="Use GPU for training")
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
            "num_workers": args.num_workers,
            "use_gpu": args.use_gpu,
            "seed": args.seed
        })
    
    # Устанавливаем seed для воспроизводимости
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    
    # Запускаем обучение
    train_model_with_ray(config)

if __name__ == "__main__":
    main()
