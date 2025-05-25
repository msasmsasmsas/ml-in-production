#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
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
import random

# ШВИДКА конфігурація для малих датасетів
CONFIG = {
    "data_path": "../crawler/downloads/",
    "image_dir": "../crawler/downloads/images/",
    "model_name": "mobilenet_v2",  # Швидка модель
    "num_epochs": 30,  # Менше епох
    "batch_size": 8,  # Менший batch для малого датасету
    "learning_rate": 0.001,  # Вищий LR для швидшого навчання
    "weight_decay": 1e-4,  # Менша регуляризація
    "dropout": 0.3,  # Менший dropout
    "seed": 42,
    "validation_split": 0.3,  # Більше валідаційних даних
    "risk_type": "diseases",
    "patience": 8,  # Менша patience
    "label_smoothing": 0.05,  # Менше smoothing

    # ВИМКНУТО для швидкості
    "progressive_resizing": False,  # Вимкнуто
    "use_mixup": False,  # Вимкнуто
    "use_cutmix": False,  # Вимкнуто
    "use_focal_loss": False,  # Вимкнуто
    "use_cosine_annealing": False,  # Вимкнуто
    "gradual_unfreezing": False,  # Вимкнуто
    "use_tta": False,  # ВИМКНУТО TTA!

    # Простіші параметри
    "image_size": 224,
    "freeze_epochs": 3,  # Заморозити backbone на 3 епохи
}

# Налаштування seed
torch.manual_seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])
random.seed(CONFIG["seed"])


class FastAgriculturalRiskDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None, risk_type="diseases"):
        """Швидкий датасет без зайвих перевірок"""
        self.image_dir = image_dir
        self.transform = transform
        self.risk_type = risk_type

        print(f"Швидке завантаження датасету: {risk_type}")

        try:
            if risk_type == "diseases":
                self.risks_df = pd.read_csv(os.path.join(csv_file, "diseases.csv"))
                self.images_df = pd.read_csv(os.path.join(csv_file, "disease_images.csv"))

            print(f"Ризиків: {len(self.risks_df)}, Зображень: {len(self.images_df)}")

        except Exception as e:
            print(f"Помилка завантаження: {e}")
            self.data = []
            self.classes = []
            self.class_to_idx = {}
            return

        self.classes = self.risks_df["name"].unique()
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        print(f"Класів: {len(self.classes)}")

        # Швидка підготовка даних
        self.data = []
        valid_images = 0

        for _, row in self.images_df.iterrows():
            try:
                risk_id = row["disease_id"]
                risk_matches = self.risks_df[self.risks_df["id"] == risk_id]

                if risk_matches.empty:
                    continue

                risk_row = risk_matches.iloc[0]
                class_name = risk_row["name"]
                image_path = row["image_path"]

                # Простіша обробка шляхів
                clean_path = image_path.replace("downloads\\images/", "").replace("downloads/images/", "")
                full_path = os.path.join(self.image_dir, clean_path)

                if os.path.exists(full_path):
                    self.data.append({
                        "image_path": full_path,
                        "class": class_name,
                        "class_idx": self.class_to_idx[class_name]
                    })
                    valid_images += 1

            except Exception:
                continue

        print(f"Валідних зображень: {valid_images}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        try:
            image = Image.open(sample["image_path"]).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, sample["class_idx"]
        except Exception:
            # Повертаємо випадкове зображення
            random_idx = random.randint(0, len(self.data) - 1)
            if random_idx != idx:
                return self.__getitem__(random_idx)
            else:
                dummy_img = torch.zeros(3, 224, 224)
                return dummy_img, sample["class_idx"]


def get_simple_transforms(image_size=224, is_training=True):
    """Прості швидкі трансформації"""
    if is_training:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


def get_fast_model(model_name, num_classes, dropout=0.3):
    """Швидка модель"""
    print(f"Створення швидкої моделі: {model_name}")

    if model_name == "mobilenet_v2":
        model = models.mobilenet_v2(pretrained=True)
        model.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(model.classifier[1].in_features, num_classes)
        )
    elif model_name == "resnet18":
        model = models.resnet18(pretrained=True)
        model.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(model.fc.in_features, num_classes)
        )
    else:
        # Default до найшвидшої моделі
        model = models.mobilenet_v2(pretrained=True)
        model.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(model.classifier[1].in_features, num_classes)
        )

    return model


def create_fast_data_loaders(config):
    """Швидкі завантажувачі даних"""
    print("Створення швидких завантажувачів...")

    train_transform = get_simple_transforms(config["image_size"], is_training=True)
    val_transform = get_simple_transforms(config["image_size"], is_training=False)

    full_dataset = FastAgriculturalRiskDataset(
        csv_file=config["data_path"],
        image_dir=config["image_dir"],
        transform=None,
        risk_type=config["risk_type"]
    )

    if len(full_dataset) == 0:
        raise ValueError("Датасет порожній!")

    # Аналіз класів
    class_counts = {}
    for sample in full_dataset.data:
        class_idx = sample["class_idx"]
        class_counts[class_idx] = class_counts.get(class_idx, 0) + 1

    # Беремо тільки класи з ≥ 2 зразками
    valid_class_indices = {class_idx for class_idx, count in class_counts.items() if count >= 2}
    valid_indices = []

    for i, sample in enumerate(full_dataset.data):
        if sample["class_idx"] in valid_class_indices:
            valid_indices.append(i)

    print(f"Після фільтрації: {len(valid_indices)} зразків")
    print(f"Валідних класів: {len(valid_class_indices)}")

    if len(valid_indices) < 10:
        print("⚠️ ДУЖЕ МАЛО ДАНИХ! Рекомендується збільшити кількість зображень")

    # ВАЖЛИВО: Перемаппінг індексів класів
    # Створюємо нове відображення: старий_індекс -> новий_індекс (0, 1, 2, ...)
    old_to_new_class_idx = {}
    new_class_names = []

    for new_idx, old_idx in enumerate(sorted(valid_class_indices)):
        old_to_new_class_idx[old_idx] = new_idx
        new_class_names.append(full_dataset.classes[old_idx])

    print(f"Перемаппінг класів: {len(old_to_new_class_idx)} класів")

    # Оновлюємо дані з новими індексами
    updated_data = []
    for i in valid_indices:
        sample = full_dataset.data[i].copy()
        old_class_idx = sample["class_idx"]
        sample["class_idx"] = old_to_new_class_idx[old_class_idx]
        updated_data.append((i, sample))

    # Простий розподіл
    train_data, val_data = train_test_split(
        updated_data,
        test_size=config["validation_split"],
        random_state=config["seed"]
    )

    train_indices = [item[0] for item in train_data]
    val_indices = [item[0] for item in val_data]

    # Створюємо датасети з оновленими індексами
    class RemappedDataset(torch.utils.data.Dataset):
        def __init__(self, base_dataset, indices, new_mapping, transform):
            self.base_dataset = base_dataset
            self.indices = indices
            self.new_mapping = new_mapping
            self.transform = transform

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, idx):
            original_idx = self.indices[idx]
            sample = self.base_dataset.data[original_idx]

            try:
                image = Image.open(sample["image_path"]).convert('RGB')
                if self.transform:
                    image = self.transform(image)

                # Використовуємо новий індекс класу
                old_class_idx = sample["class_idx"]
                new_class_idx = self.new_mapping[old_class_idx]

                return image, new_class_idx
            except Exception as e:
                print(f"Помилка завантаження {sample['image_path']}: {e}")
                dummy_img = torch.zeros(3, config["image_size"], config["image_size"])
                new_class_idx = self.new_mapping[sample["class_idx"]]
                return dummy_img, new_class_idx

    train_dataset = RemappedDataset(full_dataset, train_indices, old_to_new_class_idx, train_transform)
    val_dataset = RemappedDataset(full_dataset, val_indices, old_to_new_class_idx, val_transform)

    # Швидкі завантажувачі
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=0,
    )

    num_classes = len(valid_class_indices)
    class_names = new_class_names

    print(f"Фінальна статистика:")
    print(f"  Тренувальна вибірка: {len(train_dataset)} зразків")
    print(f"  Валідаційна вибірка: {len(val_dataset)} зразків")
    print(f"  Кількість класів: {num_classes}")
    print(f"  Батчів для навчання: {len(train_loader)}")
    print(f"  Батчів для валідації: {len(val_loader)}")
    print(f"  Класи: {class_names[:5]}..." if len(class_names) > 5 else f"  Класи: {class_names}")

    return train_loader, val_loader, num_classes, class_names


def freeze_backbone(model, model_name):
    """Заморожує backbone"""
    if model_name == "mobilenet_v2":
        for param in model.features.parameters():
            param.requires_grad = False
        print("🧊 Backbone заморожено")
    elif model_name == "resnet18":
        for name, param in model.named_parameters():
            if 'fc' not in name:
                param.requires_grad = False
        print("🧊 Backbone заморожено")


def unfreeze_backbone(model):
    """Розморожує всю модель"""
    for param in model.parameters():
        param.requires_grad = True
    print("🔥 Модель розморожена")


def train_one_epoch_fast(model, train_loader, criterion, optimizer, device, epoch):
    """Швидке навчання без зайвих аугментацій"""
    model.train()
    running_loss = 0.0
    correct_preds = 0
    total_samples = 0

    progress_bar = tqdm(train_loader, desc=f"Епоха {epoch}")

    for inputs, labels in progress_bar:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

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


def validate_fast(model, val_loader, criterion, device):
    """Швидка валідація без TTA"""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Валідація", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)

            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_loss = running_loss / len(val_loader.dataset)
    val_acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted') if len(set(all_labels)) > 1 else 0.0

    return val_loss, val_acc, f1, all_preds, all_labels


def main():
    print("⚡ ШВИДКЕ навчання для малих датасетів")

    # W&B ініціалізація
    wandb.init(project="agri-risk-classification-fast", config=CONFIG)
    config = wandb.config

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Пристрій: {device}")

    # Створюємо дані
    train_loader, val_loader, num_classes, class_names = create_fast_data_loaders(config)

    # Перевірка на мінімальну кількість даних
    if len(train_loader) < 2:
        print("❌ КРИТИЧНО МАЛО ДАНИХ!")
        print("Спробуйте зменшити batch_size або збільшити кількість зображень")
        return

    # Створюємо модель
    model = get_fast_model(config.model_name, num_classes, config.dropout)
    model = model.to(device)

    # Заморожуємо backbone спочатку
    freeze_backbone(model, config.model_name)

    # Проста функція втрат
    criterion = nn.CrossEntropyLoss(label_smoothing=config["label_smoothing"])

    # Оптимізатор
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    # Простий планувальник
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=4,
        verbose=True,
        min_lr=1e-6
    )

    best_val_f1 = 0.0
    patience_counter = 0

    os.makedirs("models", exist_ok=True)

    print(f"🚀 Починаємо швидке навчання...")

    # Навчання
    for epoch in range(config.num_epochs):
        print(f"\nЕпоха {epoch + 1}/{config.num_epochs}")

        # Розморожуємо після кількох епох
        if epoch == config["freeze_epochs"]:
            unfreeze_backbone(model)

        # Навчання
        train_loss, train_acc = train_one_epoch_fast(
            model, train_loader, criterion, optimizer, device, epoch + 1
        )

        # Валідація
        val_loss, val_acc, val_f1, all_preds, all_labels = validate_fast(
            model, val_loader, criterion, device
        )

        # Оновлюємо планувальник
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

        print(f"Train: Loss={train_loss:.3f}, Acc={train_acc:.3f}")
        print(f"Val: Loss={val_loss:.3f}, Acc={val_acc:.3f}, F1={val_f1:.3f}")

        # Early stopping
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), f"models/{config.risk_type}_{config.model_name}_best_fast.pt")
            print(f"✅ Нова найкраща модель! F1: {best_val_f1:.3f}")
        else:
            patience_counter += 1

        if patience_counter >= config.patience:
            print(f"⏹️ Early stopping після {epoch + 1} епох")
            break

    print(f"\n🎉 Швидке навчання завершено!")
    print(f"Найкращий F1: {best_val_f1:.3f}")

    # Оцінка результату
    if best_val_f1 >= 0.60:
        print("🎯 ДОБРЕ! Модель навчилася")
        print("💡 Для покращення спробуйте:")
        print("  - Збільшити кількість даних")
        print("  - Використати більші моделі")
        print("  - Додати аугментації")
    elif best_val_f1 >= 0.30:
        print("📈 Є прогрес, але потрібно більше")
        print("💡 Рекомендації:")
        print("  - Перевірте якість даних")
        print("  - Збільшіть learning rate")
        print("  - Зменшіть кількість класів")
    else:
        print("⚠️ Модель погано навчається")
        print("💡 Перевірте:")
        print("  - Правильність міток")
        print("  - Якість зображень")
        print("  - Розмір датасету")

    # Збереження метаданих
    metadata = {
        "model_name": config.model_name,
        "num_classes": num_classes,
        "class_names": class_names,
        "risk_type": config.risk_type,
        "best_val_f1": best_val_f1,
        "config": dict(config),
        "dataset_size": {
            "train": len(train_loader.dataset),
            "val": len(val_loader.dataset),
            "total": len(train_loader.dataset) + len(val_loader.dataset)
        }
    }

    with open(f"models/{config.risk_type}_{config.model_name}_metadata_fast.json", "w", encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    wandb.finish()


if __name__ == "__main__":
    main()