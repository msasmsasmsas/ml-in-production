#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd
import numpy as np
import wandb
import json
from PIL import Image, ImageFilter
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
from tqdm import tqdm
import random

# Покращена конфігурація
CONFIG = {
    "data_path": "../crawler/downloads/",
    "image_dir": "../crawler/downloads/images/",
    "model_name": "mobilenet_v2",  # Менша модель
    "num_epochs": 50,
    "batch_size": 16,  # Менший batch
    "learning_rate": 0.0001,  # Менший LR
    "weight_decay": 1e-3,  # Сильніша регуляризація
    "dropout": 0.5,
    "seed": 42,
    "validation_split": 0.25,
    "risk_type": "diseases",
    "patience": 10,  # Early stopping
    "freeze_backbone_epochs": 5,
    "label_smoothing": 0.1,
}

# Налаштування seed
torch.manual_seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])
random.seed(CONFIG["seed"])


class ImprovedAgriculturalRiskDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None, risk_type="diseases"):
        """Покращений датасет з кращою обробкою помилок"""
        self.image_dir = image_dir
        self.transform = transform
        self.risk_type = risk_type

        print(f"Завантаження датасету для типу ризику: {risk_type}")

        # Завантажуємо дані
        try:
            if risk_type == "diseases":
                self.risks_df = pd.read_csv(os.path.join(csv_file, "diseases.csv"))
                self.images_df = pd.read_csv(os.path.join(csv_file, "disease_images.csv"))
            # Додайте інші типи ризиків за потреби

            print(f"Завантажено ризиків: {len(self.risks_df)}")
            print(f"Завантажено зображень: {len(self.images_df)}")

        except Exception as e:
            print(f"Помилка завантаження CSV: {e}")
            self.data = []
            self.classes = []
            self.class_to_idx = {}
            return

        # Створюємо маппінг класів
        self.classes = self.risks_df["name"].unique()
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        print(f"Знайдено класів: {len(self.classes)}")

        # Підготовка даних з покращеною обробкою шляхів
        self.data = []
        risk_id_field = "disease_id"

        valid_images = 0
        for _, row in self.images_df.iterrows():
            try:
                risk_id = row[risk_id_field]
                risk_matches = self.risks_df[self.risks_df["id"] == risk_id]

                if risk_matches.empty:
                    continue

                risk_row = risk_matches.iloc[0]
                class_name = risk_row["name"]
                image_path = row["image_path"]

                # Покращена обробка шляхів
                clean_path = image_path.replace("downloads\\images/", "").replace("downloads/images/", "")
                clean_path = clean_path.replace("downloads\\", "").replace("downloads/", "")

                possible_paths = [
                    os.path.join(self.image_dir, clean_path),
                    os.path.join(self.image_dir, os.path.basename(image_path)),
                    image_path,
                ]

                found_path = None
                for path in possible_paths:
                    if os.path.exists(path):
                        found_path = path
                        break

                if found_path:
                    self.data.append({
                        "image_path": found_path,
                        "class": class_name,
                        "class_idx": self.class_to_idx[class_name]
                    })
                    valid_images += 1

            except Exception as e:
                continue

        print(f"Валідних зображень: {valid_images}")

        # Перевірка мінімальної кількості зразків
        if len(self.data) < 10:
            raise ValueError("Недостатньо даних для навчання!")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        image_path = sample["image_path"]

        try:
            image = Image.open(image_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, sample["class_idx"]
        except Exception as e:
            # Повертаємо випадкове зображення з датасету
            random_idx = random.randint(0, len(self.data) - 1)
            if random_idx != idx:
                return self.__getitem__(random_idx)
            else:
                dummy_img = torch.zeros(3, 224, 224)
                return dummy_img, sample["class_idx"]


def get_strong_transforms():
    """Сильні аугментації для боротьби з перенавчанням"""
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomApply([transforms.GaussianBlur(3)], p=0.3),
        transforms.RandomApply([transforms.RandomAffine(degrees=0, translate=(0.1, 0.1))], p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.1)),
    ])


def get_val_transforms():
    """Валідаційні трансформації"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def create_improved_data_loaders(config):
    """Створення покращених завантажувачів даних"""
    print("Створення покращених завантажувачів даних...")

    train_transform = get_strong_transforms()
    val_transform = get_val_transforms()

    # Створюємо повний датасет
    full_dataset = ImprovedAgriculturalRiskDataset(
        csv_file=config["data_path"],
        image_dir=config["image_dir"],
        transform=None,
        risk_type=config["risk_type"]
    )

    if len(full_dataset) == 0:
        raise ValueError("Датасет порожній!")

    print(f"Загальна кількість зразків: {len(full_dataset)}")

    # Аналіз розподілу класів
    class_counts = {}
    for sample in full_dataset.data:
        class_idx = sample["class_idx"]
        class_counts[class_idx] = class_counts.get(class_idx, 0) + 1

    print(f"Розподіл по класах (топ-10):")
    sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
    for i, (class_idx, count) in enumerate(sorted_classes[:10]):
        class_name = full_dataset.classes[class_idx]
        print(f"  {class_name}: {count} зразків")

    # Видаляємо класи з занадто малою кількістю зразків
    min_samples = 3
    valid_indices = []
    for i, sample in enumerate(full_dataset.data):
        if class_counts[sample["class_idx"]] >= min_samples:
            valid_indices.append(i)

    print(f"Після фільтрації: {len(valid_indices)} зразків")

    # Розділення на тренувальну та валідаційну вибірки
    try:
        train_indices, val_indices = train_test_split(
            valid_indices,
            test_size=config["validation_split"],
            random_state=config["seed"],
            stratify=[full_dataset.data[i]["class_idx"] for i in valid_indices]
        )
    except ValueError:
        # Якщо стратифікація неможлива
        train_indices, val_indices = train_test_split(
            valid_indices,
            test_size=config["validation_split"],
            random_state=config["seed"]
        )

    # Створюємо датасети
    train_dataset = torch.utils.data.Subset(
        ImprovedAgriculturalRiskDataset(
            csv_file=config["data_path"],
            image_dir=config["image_dir"],
            transform=train_transform,
            risk_type=config["risk_type"]
        ),
        train_indices
    )

    val_dataset = torch.utils.data.Subset(
        ImprovedAgriculturalRiskDataset(
            csv_file=config["data_path"],
            image_dir=config["image_dir"],
            transform=val_transform,
            risk_type=config["risk_type"]
        ),
        val_indices
    )

    # Створюємо завантажувачі
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=2,
        drop_last=True,  # Допомагає з регуляризацією
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=2,
    )

    # Оновлюємо кількість класів
    unique_classes = set(full_dataset.data[i]["class_idx"] for i in valid_indices)
    num_classes = len(unique_classes)
    class_names = [full_dataset.classes[i] for i in sorted(unique_classes)]

    print(f"Фінальна статистика:")
    print(f"  Тренувальна вибірка: {len(train_dataset)} зразків")
    print(f"  Валідаційна вибірка: {len(val_dataset)} зразків")
    print(f"  Кількість класів: {num_classes}")

    return train_loader, val_loader, num_classes, class_names


def get_improved_model(model_name, num_classes, dropout=0.5):
    """Створення покращеної моделі з регуляризацією"""
    print(f"Створення покращеної моделі {model_name}")

    if model_name == "mobilenet_v2":
        model = models.mobilenet_v2(pretrained=True)
        # Замінюємо класифікатор з dropout
        model.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(model.classifier[1].in_features, num_classes)
        )
    elif model_name == "resnet18":
        model = models.resnet18(pretrained=True)
        # Додаємо dropout перед фінальним шаром
        model.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(model.fc.in_features, num_classes)
        )
    else:
        raise ValueError(f"Непідтримувана модель: {model_name}")

    return model


def freeze_backbone(model, model_name):
    """Заморожує backbone моделі"""
    if model_name == "mobilenet_v2":
        for param in model.features.parameters():
            param.requires_grad = False
    elif model_name == "resnet18":
        for param in model.parameters():
            param.requires_grad = False
        # Розморожуємо тільки фінальний шар
        for param in model.fc.parameters():
            param.requires_grad = True


def unfreeze_backbone(model):
    """Розморожує всю модель"""
    for param in model.parameters():
        param.requires_grad = True


def train_one_epoch_improved(model, train_loader, criterion, optimizer, device, epoch):
    """Покращене навчання з додатковою регуляризацією"""
    model.train()
    running_loss = 0.0
    correct_preds = 0
    total_samples = 0

    progress_bar = tqdm(train_loader, desc=f"Навчання епоха {epoch}")

    for batch_idx, (inputs, labels) in enumerate(progress_bar):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # Статистика
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        correct_preds += (predicted == labels).sum().item()
        total_samples += labels.size(0)

        progress_bar.set_postfix({
            "loss": loss.item(),
            "acc": correct_preds / total_samples,
            "lr": optimizer.param_groups[0]['lr']
        })

    epoch_loss = running_loss / total_samples
    epoch_acc = correct_preds / total_samples

    return epoch_loss, epoch_acc


def validate_improved(model, val_loader, criterion, device):
    """Покращена валідація"""
    model.eval()
    running_loss = 0.0
    correct_preds = 0
    total_samples = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Валідація"):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            correct_preds += (predicted == labels).sum().item()
            total_samples += labels.size(0)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_loss = running_loss / total_samples
    val_acc = correct_preds / total_samples
    f1 = f1_score(all_labels, all_preds, average='weighted')

    return val_loss, val_acc, f1, all_preds, all_labels


def main():
    print("Початок покращеного навчання з регуляризацією")

    # W&B ініціалізація
    wandb.init(project="agri-risk-classification-improved", config=CONFIG)
    config = wandb.config

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Використовуємо пристрій: {device}")

    # Створюємо дані
    train_loader, val_loader, num_classes, class_names = create_improved_data_loaders(config)

    # Створюємо модель
    model = get_improved_model(config.model_name, num_classes, config.dropout)
    model = model.to(device)

    # Заморожуємо backbone спочатку
    if config.freeze_backbone_epochs > 0:
        freeze_backbone(model, config.model_name)
        print(f"Backbone заморожено на перші {config.freeze_backbone_epochs} епох")

    # Функція втрат з label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)

    # Оптимізатор
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    # Планувальник learning rate
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=5,
        verbose=True,
        min_lr=1e-7
    )

    # Early stopping
    best_val_f1 = 0.0
    patience_counter = 0

    os.makedirs("models", exist_ok=True)

    # Навчання
    for epoch in range(config.num_epochs):
        print(f"Епоха {epoch + 1}/{config.num_epochs}")

        # Розморожуємо backbone після певної кількості епох
        if epoch == config.freeze_backbone_epochs:
            unfreeze_backbone(model)
            print("Backbone розморожено!")

        # Навчання
        train_loss, train_acc = train_one_epoch_improved(
            model, train_loader, criterion, optimizer, device, epoch + 1
        )

        # Валідація
        val_loss, val_acc, val_f1, all_preds, all_labels = validate_improved(
            model, val_loader, criterion, device
        )

        # Оновлюємо learning rate
        scheduler.step(val_f1)

        # Логування
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_f1": val_f1,
            "lr": optimizer.param_groups[0]['lr']
        })

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")
        print(f"LR: {optimizer.param_groups[0]['lr']:.2e}")

        # Early stopping та збереження найкращої моделі
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), f"models/{config.risk_type}_{config.model_name}_best_improved.pt")
            print(f"✅ Нова найкраща модель! F1: {best_val_f1:.4f}")
        else:
            patience_counter += 1

        if patience_counter >= config.patience:
            print(f"Early stopping після {epoch + 1} епох")
            break

        print("-" * 60)

    # Фінальні результати
    print(f"\n🎉 Навчання завершено!")
    print(f"Найкращий валідаційний F1: {best_val_f1:.4f}")

    # Збереження метаданих
    metadata = {
        "model_name": config.model_name,
        "num_classes": num_classes,
        "class_names": class_names,
        "risk_type": config.risk_type,
        "best_val_f1": best_val_f1,
        "config": dict(config)
    }

    with open(f"models/{config.risk_type}_{config.model_name}_metadata_improved.json", "w", encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    wandb.finish()


if __name__ == "__main__":
    main()