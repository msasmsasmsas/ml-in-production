#!/usr/bin/env python

'''
Дистиляція знань з великої моделі до малої
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

# Налаштування логування
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('model_distillation')

class DistillationLoss(nn.Module):
    '''
    Функція втрати для дистиляції знань
    Комбінує крос-ентропію та KL-дивергенцію
    '''
    def __init__(self, alpha=0.5, temperature=2.0):
        super(DistillationLoss, self).__init__()
        self.alpha = alpha  # Вага між м'якими та жорсткими мітками
        self.temperature = temperature  # Температура для м'яких міток

    def forward(self, student_outputs, teacher_outputs, labels):
        '''
        Обчислення втрати дистиляції

        Параметри:
        -----------
        student_outputs: виходи моделі студента
        teacher_outputs: виходи моделі вчителя
        labels: справжні мітки

        Повертає:
        -----------
        комбіновану втрату
        '''
        # Жорсткі мітки: крос-ентропія з справжніми мітками
        hard_loss = F.cross_entropy(student_outputs, labels)

        # М'які мітки: KL-дивергенція між виходами студента та вчителя
        soft_student = F.log_softmax(student_outputs / self.temperature, dim=1)
        soft_teacher = F.softmax(teacher_outputs / self.temperature, dim=1)
        soft_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean') * (self.temperature ** 2)

        # Комбінована втрата
        loss = (1 - self.alpha) * hard_loss + self.alpha * soft_loss

        return loss

def load_teacher_model(model_path=None):
    '''
    Завантаження моделі вчителя (великої моделі)

    Параметри:
    -----------
    model_path: шлях до збереженої моделі (якщо None, використовується попередньо навчена модель)

    Повертає:
    -----------
    модель вчителя
    '''
    if model_path and os.path.exists(model_path):
        logger.info(f"Завантаження моделі вчителя з {model_path}")
        teacher = torch.load(model_path)
    else:
        logger.info("Завантаження попередньо навченої моделі ResNet50")
        teacher = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

    # Переведення моделі в режим оцінки
    teacher.eval()

    return teacher

def create_student_model(num_classes=1000):
    '''
    Створення моделі студента (малої моделі)

    Параметри:
    -----------
    num_classes: кількість класів для класифікації

    Повертає:
    -----------
    модель студента
    '''
    logger.info("Створення моделі студента ResNet18")
    student = resnet18(weights=None)

    # Адаптація останнього шару для кількості класів
    if student.fc.out_features != num_classes:
        in_features = student.fc.in_features
        student.fc = nn.Linear(in_features, num_classes)

    return student

def get_dataloaders(data_dir='./data', batch_size=64):
    '''
    Створення завантажувачів даних для навчання та валідації

    Параметри:
    -----------
    data_dir: директорія з даними
    batch_size: розмір батчу

    Повертає:
    -----------
    train_loader, val_loader
    '''
    # Трансформації для навчання та валідації
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

    # Створення датасетів
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
        logger.warning(f"Помилка завантаження даних: {e}. Використання CIFAR-10 для демонстрації.")

        # Запасний варіант: CIFAR-10
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

    # Створення завантажувачів даних
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
    Дистиляція знань з моделі вчителя до моделі студента

    Параметри:
    -----------
    teacher: модель вчителя
    student: модель студента
    train_loader: завантажувач даних для навчання
    val_loader: завантажувач даних для валідації
    num_epochs: кількість епох навчання
    learning_rate: швидкість навчання
    temperature: температура для м'яких міток
    alpha: вага між м'якими та жорсткими мітками
    save_path: шлях для збереження моделі

    Повертає:
    -----------
    навчена модель студента
    '''
    # Визначення пристрою для навчання
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Використання пристрою: {device}")

    # Переміщення моделей на пристрій
    teacher = teacher.to(device)
    student = student.to(device)

    # Заморожування параметрів моделі вчителя
    for param in teacher.parameters():
        param.requires_grad = False

    # Визначення оптимізатора
    optimizer = optim.SGD(student.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # Визначення функції втрати
    criterion = DistillationLoss(alpha=alpha, temperature=temperature)

    # Навчання моделі
    best_acc = 0.0
    for epoch in range(num_epochs):
        logger.info(f"Епоха {epoch+1}/{num_epochs}")

        # Навчальна фаза
        student.train()
        train_loss = 0.0
        train_acc = 0.0

        train_bar = tqdm(train_loader, desc=f"Епоха {epoch+1}/{num_epochs} [Навчання]")
        for inputs, labels in train_bar:
            inputs, labels = inputs.to(device), labels.to(device)

            # Очищення градієнтів
            optimizer.zero_grad()

            # Отримання виходів від моделей
            with torch.no_grad():
                teacher_outputs = teacher(inputs)

            student_outputs = student(inputs)

            # Обчислення втрати
            loss = criterion(student_outputs, teacher_outputs, labels)

            # Обчислення градієнтів і оновлення параметрів
            loss.backward()
            optimizer.step()

            # Статистика
            train_loss += loss.item() * inputs.size(0)

            # Обчислення точності
            _, preds = torch.max(student_outputs, 1)
            train_acc += torch.sum(preds == labels.data)

            # Оновлення прогрес-бару
            train_bar.set_postfix({'loss': loss.item()})

        # Оновлення планувальника швидкості навчання
        scheduler.step()

        # Статистика за епоху
        train_loss = train_loss / len(train_loader.dataset)
        train_acc = train_acc.float() / len(train_loader.dataset)

        # Валідаційна фаза
        student.eval()
        val_loss = 0.0
        val_acc = 0.0

        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f"Епоха {epoch+1}/{num_epochs} [Валідація]")
            for inputs, labels in val_bar:
                inputs, labels = inputs.to(device), labels.to(device)

                # Отримання виходів від моделей
                teacher_outputs = teacher(inputs)
                student_outputs = student(inputs)

                # Обчислення втрати
                loss = criterion(student_outputs, teacher_outputs, labels)

                # Статистика
                val_loss += loss.item() * inputs.size(0)

                # Обчислення точності
                _, preds = torch.max(student_outputs, 1)
                val_acc += torch.sum(preds == labels.data)

                # Оновлення прогрес-бару
                val_bar.set_postfix({'loss': loss.item()})

        # Статистика за епоху
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_acc.float() / len(val_loader.dataset)

        logger.info(f"Епоха {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Збереження найкращої моделі
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(student, save_path)
            logger.info(f"Збережено кращу модель з точністю валідації: {val_acc:.4f}")

    logger.info(f"Дистиляція завершена. Найкраща точність валідації: {best_acc:.4f}")

    # Завантаження найкращої моделі
    best_student = torch.load(save_path)

    return best_student

def main():
    parser = argparse.ArgumentParser(description="Дистиляція знань з великої моделі до малої")
    parser.add_argument("--teacher", type=str, default=None,
                        help="Шлях до моделі вчителя (якщо None, використовується попередньо навчена модель)")
    parser.add_argument("--data-dir", type=str, default="./data",
                        help="Директорія з даними")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Розмір батчу")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Кількість епох навчання")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="Швидкість навчання")
    parser.add_argument("--temperature", type=float, default=2.0,
                        help="Температура для м'яких міток")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Вага між м'якими та жорсткими мітками")
    parser.add_argument("--save-path", type=str, default="distilled_model.pt",
                        help="Шлях для збереження моделі")

    args = parser.parse_args()

    # Завантаження моделі вчителя
    teacher = load_teacher_model(args.teacher)

    # Створення моделі студента
    student = create_student_model()

    # Отримання завантажувачів даних
    train_loader, val_loader = get_dataloaders(args.data_dir, args.batch_size)

    # Дистиляція моделі
    distilled_student = distill_model(
        teacher, student, train_loader, val_loader,
        num_epochs=args.epochs, learning_rate=args.lr,
        temperature=args.temperature, alpha=args.alpha,
        save_path=args.save_path
    )

    logger.info(f"Модель успішно дистильована та збережена в {args.save_path}")

if __name__ == "__main__":
    main()
