#!/usr/bin/env python

'''
Проріджування моделі для пришвидшення інференсу
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

# Налаштування логування
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('model_pruning')

def load_model(model_path=None):
    '''
    Завантаження моделі для проріджування

    Параметри:
    -----------
    model_path: шлях до збереженої моделі (якщо None, використовується попередньо навчена модель)

    Повертає:
    -----------
    модель
    '''
    if model_path and os.path.exists(model_path):
        logger.info(f"Завантаження моделі з {model_path}")
        model = torch.load(model_path)
    else:
        logger.info("Завантаження попередньо навченої моделі ResNet50")
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

    return model

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

def evaluate_model(model, val_loader, device):
    '''
    Оцінка моделі на валідаційному наборі даних

    Параметри:
    -----------
    model: модель для оцінки
    val_loader: завантажувач даних для валідації
    device: пристрій для обчислень

    Повертає:
    -----------
    точність моделі
    '''
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Оцінка моделі"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    logger.info(f"Точність на валідаційному наборі: {accuracy:.2f}%")

    return accuracy

def measure_inference_time(model, val_loader, device, num_runs=100):
    '''
    Вимірювання часу інференсу моделі

    Параметри:
    -----------
    model: модель для вимірювання
    val_loader: завантажувач даних для валідації
    device: пристрій для обчислень
    num_runs: кількість запусків для вимірювання

    Повертає:
    -----------
    середній час інференсу
    '''
    model.eval()
    times = []

    # Отримання першого батчу для вимірювання
    for inputs, _ in val_loader:
        inputs = inputs.to(device)
        break

    # Розігрів
    with torch.no_grad():
        for _ in range(10):
            _ = model(inputs)

    # Вимірювання часу
    with torch.no_grad():
        for _ in range(num_runs):
            start_time = time.time()
            _ = model(inputs)
            end_time = time.time()
            times.append(end_time - start_time)

    # Обчислення статистики
    avg_time = sum(times) / len(times)
    std_time = np.std(times)

    logger.info(f"Середній час інференсу: {avg_time*1000:.2f} мс ± {std_time*1000:.2f} мс")

    return avg_time

def count_parameters(model):
    '''
    Підрахунок кількості параметрів моделі

    Параметри:
    -----------
    model: модель для підрахунку

    Повертає:
    -----------
    загальна кількість параметрів
    '''
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def apply_global_pruning(model, amount):
    '''
    Застосування глобального проріджування до моделі

    Параметри:
    -----------
    model: модель для проріджування
    amount: відсоток ваг для видалення (0-1)

    Повертає:
    -----------
    проріджена модель
    '''
    logger.info(f"Застосування глобального проріджування з коефіцієнтом {amount}")

    # Отримання всіх параметрів для проріджування
    parameters_to_prune = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            parameters_to_prune.append((module, 'weight'))

    # Застосування глобального проріджування
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=amount,
    )

    return model

def apply_local_pruning(model, amount):
    '''
    Застосування локального проріджування до моделі

    Параметри:
    -----------
    model: модель для проріджування
    amount: відсоток ваг для видалення (0-1)

    Повертає:
    -----------
    проріджена модель
    '''
    logger.info(f"Застосування локального проріджування з коефіцієнтом {amount}")

    # Проріджування шарів
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            prune.l1_unstructured(module, name='weight', amount=amount)
        elif isinstance(module, nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=amount)

    return model

def fine_tune_pruned_model(model, train_loader, val_loader, num_epochs=5, learning_rate=0.001):
    '''
    Донавчання проріджуваної моделі

    Параметри:
    -----------
    model: проріджена модель
    train_loader: завантажувач даних для навчання
    val_loader: завантажувач даних для валідації
    num_epochs: кількість епох навчання
    learning_rate: швидкість навчання

    Повертає:
    -----------
    донавчена модель
    '''
    logger.info("Донавчання проріджуваної моделі")

    # Визначення пристрою
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Визначення оптимізатора та функції втрати
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # Навчання моделі
    best_acc = 0.0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        train_bar = tqdm(train_loader, desc=f"Епоха {epoch+1}/{num_epochs}")
        for inputs, labels in train_bar:
            inputs, labels = inputs.to(device), labels.to(device)

            # Очищення градієнтів
            optimizer.zero_grad()

            # Прямий прохід
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Зворотній прохід та оптимізація
            loss.backward()
            optimizer.step()

            # Статистика
            running_loss += loss.item() * inputs.size(0)
            train_bar.set_postfix({'loss': loss.item()})

        # Оновлення планувальника
        scheduler.step()

        # Оцінка моделі
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
        logger.info(f"Епоха {epoch+1}/{num_epochs} - Втрата: {running_loss/len(train_loader.dataset):.4f}, Точність: {accuracy:.2f}%")

        # Збереження найкращої моделі
        if accuracy > best_acc:
            best_acc = accuracy
            torch.save(model, 'pruned_model_best.pt')

    # Завантаження найкращої моделі
    model = torch.load('pruned_model_best.pt')

    return model

def remove_pruning_masks(model):
    '''
    Видалення масок проріджування для постійного ефекту

    Параметри:
    -----------
    model: проріджена модель

    Повертає:
    -----------
    модель з видаленими масками проріджування
    '''
    logger.info("Видалення масок проріджування")

    # Видалення масок
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            try:
                prune.remove(module, 'weight')
            except:
                pass

    return model

def main():
    parser = argparse.ArgumentParser(description="Проріджування моделі для пришвидшення інференсу")
    parser.add_argument("--model", type=str, default=None,
                        help="Шлях до моделі (якщо None, використовується попередньо навчена модель)")
    parser.add_argument("--data-dir", type=str, default="./data",
                        help="Директорія з даними")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Розмір батчу")
    parser.add_argument("--pruning-method", type=str, choices=["global", "local"], default="global",
                        help="Метод проріджування (global або local)")
    parser.add_argument("--amount", type=float, default=0.3,
                        help="Відсоток ваг для видалення (0-1)")
    parser.add_argument("--fine-tune", action="store_true",
                        help="Донавчання після проріджування")
    parser.add_argument("--fine-tune-epochs", type=int, default=5,
                        help="Кількість епох донавчання")
    parser.add_argument("--fine-tune-lr", type=float, default=0.001,
                        help="Швидкість навчання для донавчання")
    parser.add_argument("--save-path", type=str, default="pruned_model.pt",
                        help="Шлях для збереження моделі")

    args = parser.parse_args()

    # Завантаження моделі
    model = load_model(args.model)

    # Визначення пристрою
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Отримання завантажувачів даних
    train_loader, val_loader = get_dataloaders(args.data_dir, args.batch_size)

    # Оцінка моделі перед проріджуванням
    logger.info("Оцінка моделі перед проріджуванням")
    initial_parameters = count_parameters(model)
    logger.info(f"Кількість параметрів: {initial_parameters:,}")

    initial_accuracy = evaluate_model(model, val_loader, device)
    initial_inference_time = measure_inference_time(model, val_loader, device)

    # Проріджування моделі
    if args.pruning_method == "global":
        pruned_model = apply_global_pruning(model, args.amount)
    else:
        pruned_model = apply_local_pruning(model, args.amount)

    # Оцінка моделі після проріджування
    logger.info("Оцінка моделі після проріджування")
    pruned_accuracy = evaluate_model(pruned_model, val_loader, device)
    pruned_inference_time = measure_inference_time(pruned_model, val_loader, device)

    # Донавчання моделі, якщо потрібно
    if args.fine_tune:
        pruned_model = fine_tune_pruned_model(
            pruned_model, train_loader, val_loader,
            num_epochs=args.fine_tune_epochs,
            learning_rate=args.fine_tune_lr
        )

        # Оцінка моделі після донавчання
        logger.info("Оцінка моделі після донавчання")
        fine_tuned_accuracy = evaluate_model(pruned_model, val_loader, device)
        fine_tuned_inference_time = measure_inference_time(pruned_model, val_loader, device)

    # Видалення масок проріджування
    pruned_model = remove_pruning_masks(pruned_model)

    # Збереження моделі
    torch.save(pruned_model, args.save_path)
    logger.info(f"Проріджена модель збережена в {args.save_path}")

    # Виведення результатів
    logger.info("\nРезультати проріджування:")
    logger.info(f"Початкова кількість параметрів: {initial_parameters:,}")
    logger.info(f"Початкова точність: {initial_accuracy:.2f}%")
    logger.info(f"Початковий час інференсу: {initial_inference_time*1000:.2f} мс")

    final_parameters = count_parameters(pruned_model)
    logger.info(f"\nФінальна кількість параметрів: {final_parameters:,} ({final_parameters/initial_parameters*100:.2f}% від початкової)")
    logger.info(f"Точність після проріджування: {pruned_accuracy:.2f}% ({pruned_accuracy-initial_accuracy:+.2f}%)")
    logger.info(f"Час інференсу після проріджування: {pruned_inference_time*1000:.2f} мс ({(1-pruned_inference_time/initial_inference_time)*100:.2f}% пришвидшення)")

    if args.fine_tune:
        logger.info(f"\nТочність після донавчання: {fine_tuned_accuracy:.2f}% ({fine_tuned_accuracy-initial_accuracy:+.2f}%)")
        logger.info(f"Час інференсу після донавчання: {fine_tuned_inference_time*1000:.2f} мс ({(1-fine_tuned_inference_time/initial_inference_time)*100:.2f}% пришвидшення)")

if __name__ == "__main__":
    main()
