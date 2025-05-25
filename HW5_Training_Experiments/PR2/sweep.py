#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import wandb
import json
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import random

# Конфігурація sweep для пошуку гіперпараметрів (ВИПРАВЛЕНА)
SWEEP_CONFIG = {
    "method": "bayes",  # random, grid, bayes
    "metric": {
        "name": "val_f1",
        "goal": "maximize"
    },
    "parameters": {
        "model_name": {
            "values": ["resnet18", "resnet34", "mobilenet_v2"]
        },
        "batch_size": {
            "values": [8, 16, 32]
        },
        # ВИПРАВЛЕНО: використовуємо log_uniform_values замість log_uniform
        "learning_rate": {
            "min": 0.0001,  # 1e-4
            "max": 0.005,  # 5e-3
            "distribution": "log_uniform_values"
        },
        "weight_decay": {
            "min": 0.00001,  # 1e-5
            "max": 0.01,  # 1e-2
            "distribution": "log_uniform_values"
        },
        "dropout": {
            "min": 0.1,
            "max": 0.6,
            "distribution": "uniform"
        },
        "num_epochs": {
            "values": [15, 20, 25]
        },
        "validation_split": {
            "values": [0.2, 0.25, 0.3]
        }
    }
}


class OptimizedDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None, risk_type="diseases"):
        """Оптимізований датасет з виправленням шляхів"""
        self.image_dir = image_dir
        self.transform = transform

        print(f"🔄 Завантаження даних: {risk_type}")

        # Завантажуємо CSV файли
        if risk_type == "diseases":
            self.risks_df = pd.read_csv(os.path.join(csv_file, "diseases.csv"))
            self.images_df = pd.read_csv(os.path.join(csv_file, "disease_images.csv"))
            risk_id_field = "disease_id"
        elif risk_type == "pests":
            self.risks_df = pd.read_csv(os.path.join(csv_file, "vermins.csv"))
            self.images_df = pd.read_csv(os.path.join(csv_file, "vermin_images.csv"))
            risk_id_field = "vermin_id"
        elif risk_type == "weeds":
            self.risks_df = pd.read_csv(os.path.join(csv_file, "weeds.csv"))
            self.images_df = pd.read_csv(os.path.join(csv_file, "weed_images.csv"))
            risk_id_field = "weed_id"

        print(f"📊 Ризиків: {len(self.risks_df)}, Зображень: {len(self.images_df)}")

        # Збираємо дані з виправленням шляхів
        self.data = []
        self.class_names = []

        for _, row in self.images_df.iterrows():
            try:
                risk_id = row[risk_id_field]
                risk_info = self.risks_df[self.risks_df["id"] == risk_id]

                if risk_info.empty:
                    continue

                class_name = risk_info.iloc[0]["name"]
                image_path = row["image_path"]

                # Виправлення шляхів (як у робочій версії)
                clean_path = image_path.replace("downloads\\images/", "").replace("downloads/images/", "")
                clean_path = clean_path.replace("downloads\\", "").replace("downloads/", "")
                full_path = os.path.join(self.image_dir, clean_path)

                if os.path.exists(full_path):
                    self.data.append({
                        "image_path": full_path,
                        "class_name": class_name
                    })

                    if class_name not in self.class_names:
                        self.class_names.append(class_name)

            except Exception as e:
                continue

        # Створення маппінгу класів
        self.class_names = sorted(list(set(self.class_names)))
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}

        # Фільтрація класів з малою кількістю зразків
        class_counts = {}
        for item in self.data:
            class_name = item["class_name"]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1

        min_samples = 2
        filtered_data = []
        valid_classes = {name for name, count in class_counts.items() if count >= min_samples}

        for item in self.data:
            if item["class_name"] in valid_classes:
                item["class_idx"] = self.class_to_idx[item["class_name"]]
                filtered_data.append(item)

        self.data = filtered_data
        self.class_names = sorted(list(valid_classes))
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}

        # Оновлюємо індекси
        for item in self.data:
            item["class_idx"] = self.class_to_idx[item["class_name"]]

        print(f"✅ Після фільтрації: {len(self.data)} зображень, {len(self.class_names)} класів")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        try:
            image = Image.open(item["image_path"]).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, item["class_idx"]
        except Exception as e:
            # Повертаємо випадкове зображення при помилці
            random_idx = random.randint(0, len(self.data) - 1)
            if random_idx != idx:
                return self.__getitem__(random_idx)
            else:
                dummy_img = torch.zeros(3, 224, 224)
                return dummy_img, item["class_idx"]


def get_transforms(config):
    """Створення трансформацій з урахуванням конфігурації"""

    # Базові трансформації
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    return train_transform, val_transform


def create_optimized_loaders(config):
    """Створення оптимізованих завантажувачів даних"""
    print("🔄 Створення завантажувачів даних...")

    train_transform, val_transform = get_transforms(config)

    # Створюємо повний датасет
    dataset = OptimizedDataset(
        csv_file=config["data_path"],
        image_dir=config["image_dir"],
        transform=None,
        risk_type=config.get("risk_type", "diseases")
    )

    if len(dataset.data) < 10:
        raise ValueError(f"❌ Недостатньо даних: {len(dataset.data)} зображень")

    # Розділяємо дані
    train_data, val_data = train_test_split(
        dataset.data,
        test_size=config.get("validation_split", 0.25),
        random_state=config.get("seed", 42)
    )

    print(f"📊 Розділення: {len(train_data)} тренувальних, {len(val_data)} валідаційних")

    # Створюємо підмножини
    class OptimizedSubset(Dataset):
        def __init__(self, data, class_to_idx, transform):
            self.data = data
            self.class_to_idx = class_to_idx
            self.transform = transform

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            item = self.data[idx]
            try:
                image = Image.open(item["image_path"]).convert('RGB')
                if self.transform:
                    image = self.transform(image)
                return image, item["class_idx"]
            except:
                dummy_img = torch.zeros(3, 224, 224)
                return dummy_img, item["class_idx"]

    train_dataset = OptimizedSubset(train_data, dataset.class_to_idx, train_transform)
    val_dataset = OptimizedSubset(val_data, dataset.class_to_idx, val_transform)

    # Створюємо завантажувачі
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get("batch_size", 16),
        shuffle=True,
        num_workers=0,  # Для CPU
        drop_last=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.get("batch_size", 16),
        shuffle=False,
        num_workers=0
    )

    num_classes = len(dataset.class_names)

    print(f"✅ Готово! Класів: {num_classes}, Батчів: train={len(train_loader)}, val={len(val_loader)}")

    return train_loader, val_loader, num_classes, dataset.class_names


def get_optimized_model(model_name, num_classes, dropout=0.3):
    """Створення моделі з урахуванням dropout"""
    print(f"🧠 Створення моделі {model_name} для {num_classes} класів")

    if model_name == "resnet18":
        model = models.resnet18(weights='IMAGENET1K_V1')
        model.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(model.fc.in_features, num_classes)
        )
    elif model_name == "resnet34":
        model = models.resnet34(weights='IMAGENET1K_V1')
        model.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(model.fc.in_features, num_classes)
        )
    elif model_name == "mobilenet_v2":
        model = models.mobilenet_v2(weights='IMAGENET1K_V1')
        model.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(model.classifier[1].in_features, num_classes)
        )
    else:
        # За замовчуванням MobileNetV2
        print(f"⚠️ Невідома модель {model_name}, використовуємо mobilenet_v2")
        model = models.mobilenet_v2(weights='IMAGENET1K_V1')
        model.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(model.classifier[1].in_features, num_classes)
        )

    return model


def train_epoch_optimized(model, loader, criterion, optimizer, device):
    """Оптимізоване навчання однієї епохи"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for inputs, labels in tqdm(loader, desc="Навчання", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    return total_loss / len(loader), correct / total


def validate_optimized(model, loader, criterion, device):
    """Оптимізована валідація"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Валідація", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_loss = total_loss / len(loader)
    val_acc = accuracy_score(all_labels, all_preds)
    val_f1 = f1_score(all_labels, all_preds, average='weighted')

    return val_loss, val_acc, val_f1


def train_model_sweep():
    """Функція навчання для sweep"""

    # Ініціалізація W&B з ПРАВИЛЬНИМ проектом
    with wandb.init(project="agri-risk-hyperparameter-search") as run:
        config = wandb.config

        # Додаємо базові параметри
        config_dict = dict(config)
        config_dict.update({
            "data_path": "../crawler/downloads/",
            "image_dir": "../crawler/downloads/images/",
            "risk_type": "diseases",
            "seed": 42
        })

        # Налаштування seed
        torch.manual_seed(config_dict["seed"])
        np.random.seed(config_dict["seed"])
        random.seed(config_dict["seed"])

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"💻 Пристрій: {device}")

        try:
            # Дані
            train_loader, val_loader, num_classes, class_names = create_optimized_loaders(config_dict)

            if num_classes < 2:
                print(f"❌ Недостатньо класів: {num_classes}")
                wandb.log({"val_f1": 0.0})
                return

            # Модель
            model = get_optimized_model(
                config.model_name,
                num_classes,
                config.get("dropout", 0.3)
            )
            model = model.to(device)

            # Заморожування backbone на перші епохи
            freeze_epochs = 3
            if hasattr(model, 'features'):  # MobileNet
                for param in model.features.parameters():
                    param.requires_grad = False
            else:  # ResNet
                for name, param in model.named_parameters():
                    if 'fc' not in name:
                        param.requires_grad = False

            print("🧊 Backbone заморожено")

            # Навчання
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.get("weight_decay", 1e-4)
            )

            best_f1 = 0.0
            patience_counter = 0
            patience = 5

            num_epochs = config.get("num_epochs", 20)

            print(f"🚀 Початок навчання на {num_epochs} епох")

            for epoch in range(num_epochs):
                # Розморожуємо після freeze_epochs
                if epoch == freeze_epochs:
                    for param in model.parameters():
                        param.requires_grad = True
                    print("🔥 Модель розморожена")

                # Навчання та валідація
                train_loss, train_acc = train_epoch_optimized(model, train_loader, criterion, optimizer, device)
                val_loss, val_acc, val_f1 = validate_optimized(model, val_loader, criterion, device)

                # Логування
                wandb.log({
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "val_f1": val_f1,
                })

                print(f"Епоха {epoch + 1}: Train Acc={train_acc:.3f}, Val F1={val_f1:.3f}")

                # Early stopping
                if val_f1 > best_f1:
                    best_f1 = val_f1
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    print(f"⏹️ Early stopping після {epoch + 1} епох")
                    break

            print(f"🎉 Найкращий F1: {best_f1:.3f}")

            # Фінальне логування
            wandb.log({"final_f1": best_f1})

        except Exception as e:
            print(f"❌ Помилка: {e}")
            wandb.log({"val_f1": 0.0})


def main():
    parser = argparse.ArgumentParser(description="W&B Sweeps для пошуку гіперпараметрів")
    parser.add_argument("--create", action="store_true", help="Створити новий sweep")
    parser.add_argument("--agent", type=str, help="Запустити агент з вказаним sweep ID")
    parser.add_argument("--count", type=int, default=15, help="Кількість запусків для агента")

    args = parser.parse_args()

    # ВАЖЛИВО: Явно вказуємо проект для всіх операцій
    PROJECT_NAME = "agri-risk-hyperparameter-search"

    if args.create:
        # Створення нового sweep
        sweep_id = wandb.sweep(SWEEP_CONFIG, project=PROJECT_NAME)
        print(f"🆔 Створено sweep: {sweep_id}")
        print(f"🌐 URL: https://wandb.ai/msas-agrichain/{PROJECT_NAME}/sweeps/{sweep_id}")
        print(f"🚀 Для запуску агента: python sweep.py --agent {sweep_id}")

    elif args.agent:
        # Запуск агента з ЯВНИМ вказанням проекту
        print(f"🤖 Запуск агента для sweep {args.agent}")
        print(f"📊 Кількість експериментів: {args.count}")
        print(f"📁 Проект: {PROJECT_NAME}")

        # ВИПРАВЛЕННЯ: додаємо project до wandb.agent
        wandb.agent(
            args.agent,
            function=train_model_sweep,
            count=args.count,
            project=PROJECT_NAME  # ✅ ДОДАНО!
        )

    else:
        # За замовчуванням створюємо sweep та запускаємо
        print(f"🆔 Створення sweep у проекті: {PROJECT_NAME}")
        sweep_id = wandb.sweep(SWEEP_CONFIG, project=PROJECT_NAME)
        print(f"🆔 Створено sweep: {sweep_id}")
        print(f"🚀 Запуск експериментів...")

        # ВИПРАВЛЕННЯ: додаємо project до wandb.agent
        wandb.agent(
            sweep_id,
            function=train_model_sweep,
            count=10,
            project=PROJECT_NAME  # ✅ ДОДАНО!
        )


if __name__ == "__main__":
    main()