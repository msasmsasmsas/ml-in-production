#!/usr/bin/env python

'''
Квантизація моделі для пришвидшення інференсу
'''

import os
import time
import logging
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.quantization
from torch.quantization import quantize_dynamic, quantize_static, QuantStub, DeQuantStub
from torch.utils.data import DataLoader, Dataset

import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights

# Налаштування логування
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('model_quantization')

class QuantizableResNet(nn.Module):
    '''
    Обгортка для ResNet з підтримкою квантизації
    '''
    def __init__(self, model):
        super(QuantizableResNet, self).__init__()
        self.quant = QuantStub()
        self.model = model
        self.dequant = DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.model(x)
        x = self.dequant(x)
        return x

    def fuse_model(self):
        '''
        Злиття операцій Conv+BN+ReLU для пришвидшення інференсу
        '''
        for module_name, module in self.model.named_children():
            if "layer" in module_name:
                for basic_block_name, basic_block in module.named_children():
                    torch.quantization.fuse_modules(
                        basic_block, ["conv1", "bn1", "relu"], inplace=True
                    )
                    torch.quantization.fuse_modules(
                        basic_block, ["conv2", "bn2"], inplace=True
                    )

        logger.info("Модель успішно підготовлена до квантизації")

def load_model(model_path=None):
    '''
    Завантаження моделі для квантизації

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

    # Трансформації для квантизації (без нормалізації для int8)
    quantization_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
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

        calib_dataset = torchvision.datasets.ImageFolder(
            os.path.join(data_dir, 'val'),
            transform=quantization_transform
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

        calib_dataset = torchvision.datasets.CIFAR10(
            root='./data',
            train=False,
            download=True,
            transform=quantization_transform
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

    # Створення завантажувача для калібрування (менший розмір)
    calib_loader = DataLoader(
        calib_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    return train_loader, val_loader, calib_loader

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

def get_model_size(model):
    '''
    Отримання розміру моделі в МБ

    Параметри:
    -----------
    model: модель для вимірювання

    Повертає:
    -----------
    розмір моделі в МБ
    '''
    torch.save(model.state_dict(), "temp_model.pt")
    size_mb = os.path.getsize("temp_model.pt") / (1024 * 1024)
    os.remove("temp_model.pt")
    return size_mb

def calibrate_model(model, calib_loader, device):
    '''
    Калібрування моделі для статичної квантизації

    Параметри:
    -----------
    model: модель для калібрування
    calib_loader: завантажувач даних для калібрування
    device: пристрій для обчислень

    Повертає:
    -----------
    відкалібрована модель
    '''
    logger.info("Калібрування моделі для статичної квантизації")
    model.eval()

    # Налаштування калібрування
    with torch.no_grad():
        # Обмежуємось меншою кількістю батчів для швидшого калібрування
        for i, (inputs, _) in enumerate(tqdm(calib_loader, desc="Калібрування")):
            inputs = inputs.to(device)
            model(inputs)
            if i >= 10:  # Зазвичай достатньо 10-20 батчів
                break

    return model

def dynamic_quantize_model(model):
    '''
    Динамічна квантизація моделі

    Параметри:
    -----------
    model: модель для квантизації

    Повертає:
    -----------
    квантизована модель
    '''
    logger.info("Застосування динамічної квантизації")

    # Динамічна квантизація лінійних шарів
    quantized_model = torch.quantization.quantize_dynamic(
        model, 
        {nn.Linear}, 
        dtype=torch.qint8
    )

    return quantized_model

def static_quantize_model(model, calib_loader):
    '''
    Статична квантизація моделі

    Параметри:
    -----------
    model: модель для квантизації
    calib_loader: завантажувач даних для калібрування

    Повертає:
    -----------
    квантизована модель
    '''
    logger.info("Підготовка до статичної квантизації")

    # Підготовка моделі з QuantStub та DeQuantStub
    quantizable_model = QuantizableResNet(model)

    # Злиття шарів Conv+BN+ReLU
    quantizable_model.fuse_model()

    # Переведення моделі в режим оцінки
    quantizable_model.eval()

    # Встановлення конфігурації квантизації
    quantizable_model.qconfig = torch.quantization.get_default_qconfig('fbgemm')

    # Підготовка моделі до квантизації
    torch.quantization.prepare(quantizable_model, inplace=True)

    # Калібрування моделі
    with torch.no_grad():
        for i, (inputs, _) in enumerate(tqdm(calib_loader, desc="Калібрування")):
            quantizable_model(inputs)
            if i >= 10:  # Обмежуємось для швидшого калібрування
                break

    # Конвертація в квантизовану модель
    torch.quantization.convert(quantizable_model, inplace=True)

    logger.info("Статична квантизація завершена")

    return quantizable_model

def main():
    parser = argparse.ArgumentParser(description="Квантизація моделі для пришвидшення інференсу")
    parser.add_argument("--model", type=str, default=None,
                        help="Шлях до моделі (якщо None, використовується попередньо навчена модель)")
    parser.add_argument("--data-dir", type=str, default="./data",
                        help="Директорія з даними")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Розмір батчу")
    parser.add_argument("--quantization-method", type=str, choices=["dynamic", "static"], default="dynamic",
                        help="Метод квантизації (dynamic або static)")
    parser.add_argument("--save-path", type=str, default="quantized_model.pt",
                        help="Шлях для збереження моделі")

    args = parser.parse_args()

    # Завантаження моделі
    model = load_model(args.model)

    # Визначення пристрою для початкової оцінки
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Отримання завантажувачів даних
    train_loader, val_loader, calib_loader = get_dataloaders(args.data_dir, args.batch_size)

    # Оцінка моделі перед квантизацією
    logger.info("Оцінка моделі перед квантизацією")
    initial_size = get_model_size(model)
    initial_accuracy = evaluate_model(model, val_loader, device)
    initial_inference_time = measure_inference_time(model, val_loader, device)

    # Квантизація виконується на CPU
    cpu_device = torch.device("cpu")
    model = model.to(cpu_device)

    # Квантизація моделі
    if args.quantization_method == "dynamic":
        quantized_model = dynamic_quantize_model(model)
    else:  # static
        quantized_model = static_quantize_model(model, calib_loader)

    # Оцінка квантизованої моделі
    logger.info("Оцінка квантизованої моделі")
    quantized_size = get_model_size(quantized_model)
    quantized_accuracy = evaluate_model(quantized_model, val_loader, cpu_device)
    quantized_inference_time = measure_inference_time(quantized_model, val_loader, cpu_device)

    # Збереження моделі
    torch.save(quantized_model, args.save_path)
    logger.info(f"Квантизована модель збережена в {args.save_path}")

    # Виведення результатів
    logger.info("\nРезультати квантизації:")
    logger.info(f"Початковий розмір моделі: {initial_size:.2f} МБ")
    logger.info(f"Початкова точність: {initial_accuracy:.2f}%")
    logger.info(f"Початковий час інференсу: {initial_inference_time*1000:.2f} мс")

    logger.info(f"\nРозмір квантизованої моделі: {quantized_size:.2f} МБ ({quantized_size/initial_size*100:.2f}% від початкової)")
    logger.info(f"Точність квантизованої моделі: {quantized_accuracy:.2f}% ({quantized_accuracy-initial_accuracy:+.2f}%)")
    logger.info(f"Час інференсу квантизованої моделі: {quantized_inference_time*1000:.2f} мс ({(1-quantized_inference_time/initial_inference_time)*100:.2f}% пришвидшення)")

if __name__ == "__main__":
    main()
