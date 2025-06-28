#!/usr/bin/env python

'''
Бенчмаркінг моделей машинного навчання після оптимізації
'''

import os
import time
import json
import logging
import argparse
from datetime import datetime
from tabulate import tabulate
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights

# Налаштування логування
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('benchmark_models')

class ModelBenchmark:
    '''
    Клас для бенчмаркінгу моделей машинного навчання
    '''
    def __init__(self, models_dict, dataset_path=None, batch_size=32, num_runs=100, save_dir='benchmark_results'):
        '''
        Ініціалізація бенчмаркера

        Параметри:
        -----------
        models_dict: словник з моделями для бенчмаркінгу {"name": model}
        dataset_path: шлях до датасету для тестування
        batch_size: розмір батчу для інференсу
        num_runs: кількість запусків для вимірювання
        save_dir: директорія для збереження результатів
        '''
        self.models_dict = models_dict
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.num_runs = num_runs
        self.save_dir = save_dir

        # Створення директорії для результатів, якщо її немає
        os.makedirs(save_dir, exist_ok=True)

        # Результати бенчмаркінгу
        self.results = {}

        # Завантаження даних для тестування
        self.test_loader = self._load_test_data()

    def _load_test_data(self):
        '''
        Завантаження даних для тестування

        Повертає:
        -----------
        завантажувач даних
        '''
        # Трансформації для тестового набору
        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # Спроба завантажити датасет
        try:
            if self.dataset_path and os.path.exists(self.dataset_path):
                test_dataset = torchvision.datasets.ImageFolder(
                    self.dataset_path,
                    transform=test_transform
                )
            else:
                # Використання CIFAR-10 як запасного варіанту
                logger.info("Використання CIFAR-10 як тестового датасету")
                test_dataset = torchvision.datasets.CIFAR10(
                    root='./data',
                    train=False,
                    download=True,
                    transform=test_transform
                )
        except Exception as e:
            logger.warning(f"Помилка завантаження даних: {e}. Використання CIFAR-10.")
            test_dataset = torchvision.datasets.CIFAR10(
                root='./data',
                train=False,
                download=True,
                transform=test_transform
            )

        # Створення завантажувача даних
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

        return test_loader

    def _measure_model_size(self, model):
        '''
        Вимірювання розміру моделі

        Параметри:
        -----------
        model: модель для вимірювання

        Повертає:
        -----------
        розмір моделі в МБ
        '''
        # Збереження моделі у тимчасовий файл
        temp_file = os.path.join(self.save_dir, "temp_model.pt")
        torch.save(model.state_dict(), temp_file)

        # Отримання розміру файлу
        size_mb = os.path.getsize(temp_file) / (1024 * 1024)

        # Видалення тимчасового файлу
        os.remove(temp_file)

        return size_mb

    def _count_parameters(self, model):
        '''
        Підрахунок кількості параметрів моделі

        Параметри:
        -----------
        model: модель для підрахунку

        Повертає:
        -----------
        кількість параметрів
        '''
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def _evaluate_accuracy(self, model, device):
        '''
        Оцінка точності моделі

        Параметри:
        -----------
        model: модель для оцінки
        device: пристрій для обчислень

        Повертає:
        -----------
        точність моделі у відсотках
        '''
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in tqdm(self.test_loader, desc="Оцінка точності"):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        return accuracy

    def _measure_inference_time(self, model, device):
        '''
        Вимірювання часу інференсу моделі

        Параметри:
        -----------
        model: модель для вимірювання
        device: пристрій для обчислень

        Повертає:
        -----------
        словник з часом інференсу (середній, мін, макс, стд)
        '''
        model.eval()
        times = []
        batch_times = []

        # Отримання першого батчу для вимірювання одиночного інференсу
        single_input = None
        for inputs, _ in self.test_loader:
            single_input = inputs[0:1].to(device)  # Один зразок
            inputs = inputs.to(device)             # Повний батч
            break

        # Розігрів
        with torch.no_grad():
            for _ in range(10):
                _ = model(single_input)
                _ = model(inputs)

        # Вимірювання часу інференсу одиночного зразка
        with torch.no_grad():
            for _ in range(self.num_runs):
                start_time = time.time()
                _ = model(single_input)
                end_time = time.time()
                times.append(end_time - start_time)

        # Вимірювання часу інференсу батчу
        with torch.no_grad():
            for _ in range(self.num_runs // 10):  # Менше запусків для батчу
                start_time = time.time()
                _ = model(inputs)
                end_time = time.time()
                batch_times.append(end_time - start_time)

        # Обчислення статистики
        time_stats = {
            "single": {
                "mean": float(np.mean(times) * 1000),  # мс
                "min": float(np.min(times) * 1000),    # мс
                "max": float(np.max(times) * 1000),    # мс
                "std": float(np.std(times) * 1000)     # мс
            },
            "batch": {
                "mean": float(np.mean(batch_times) * 1000),  # мс
                "min": float(np.min(batch_times) * 1000),    # мс
                "max": float(np.max(batch_times) * 1000),    # мс
                "std": float(np.std(batch_times) * 1000),    # мс
                "batch_size": self.batch_size
            }
        }

        return time_stats

    def _measure_memory_usage(self, model, device):
        '''
        Вимірювання використання пам'яті моделлю

        Параметри:
        -----------
        model: модель для вимірювання
        device: пристрій для обчислень

        Повертає:
        -----------
        використання пам'яті в МБ
        '''
        # Очищення кешу CUDA
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        # Отримання батчу для інференсу
        for inputs, _ in self.test_loader:
            inputs = inputs.to(device)
            break

        # Інференс для вимірювання пам'яті
        with torch.no_grad():
            _ = model(inputs)

        # Вимірювання використання пам'яті
        if device.type == 'cuda':
            memory_usage = torch.cuda.max_memory_allocated() / (1024 * 1024)  # МБ
        else:
            # Приблизна оцінка для CPU (не точна)
            memory_usage = self._measure_model_size(model) * 2

        return float(memory_usage)

    def _profile_model(self, model, device):
        '''
        Профілювання моделі за допомогою torch.profiler

        Параметри:
        -----------
        model: модель для профілювання
        device: пристрій для обчислень

        Повертає:
        -----------
        словник з результатами профілювання
        '''
        try:
            from torch.profiler import profile, record_function, ProfilerActivity

            # Отримання батчу для інференсу
            for inputs, _ in self.test_loader:
                inputs = inputs.to(device)
                break

            # Розігрів
            with torch.no_grad():
                for _ in range(5):
                    _ = model(inputs)

            # Профілювання
            activities = [ProfilerActivity.CPU]
            if device.type == 'cuda':
                activities.append(ProfilerActivity.CUDA)

            with torch.no_grad():
                with profile(activities=activities, record_shapes=True) as prof:
                    with record_function("model_inference"):
                        _ = model(inputs)

            # Аналіз результатів профілювання
            total_time = prof.self_cpu_time_total / 1000.0  # мс
            table = prof.key_averages().table(sort_by="cpu_time_total", row_limit=10)

            # Отримання топ-5 найдовших операцій
            top_ops = []
            for row in prof.key_averages().table(sort_by="cpu_time_total", row_limit=5).split("\n")[2:-1]:
                parts = row.split()
                if len(parts) >= 7:
                    op_name = parts[6]
                    cpu_time = float(parts[1])
                    top_ops.append({"name": op_name, "cpu_time": cpu_time})

            return {
                "total_time": float(total_time),
                "top_ops": top_ops,
                "profile_table": table
            }

        except Exception as e:
            logger.warning(f"Помилка профілювання: {e}")
            return {"error": str(e)}

    def benchmark_model(self, model_name):
        '''
        Бенчмаркінг моделі

        Параметри:
        -----------
        model_name: назва моделі у словнику моделей

        Повертає:
        -----------
        словник з результатами бенчмаркінгу
        '''
        if model_name not in self.models_dict:
            logger.error(f"Модель {model_name} не знайдена у словнику моделей")
            return None

        model = self.models_dict[model_name]
        logger.info(f"Бенчмаркінг моделі: {model_name}")

        # Визначення пристрою
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        # Вимірювання основних метрик
        model_size = self._measure_model_size(model)
        logger.info(f"Розмір моделі: {model_size:.2f} МБ")

        num_params = self._count_parameters(model)
        logger.info(f"Кількість параметрів: {num_params:,}")

        memory_usage = self._measure_memory_usage(model, device)
        logger.info(f"Використання пам'яті: {memory_usage:.2f} МБ")

        accuracy = self._evaluate_accuracy(model, device)
        logger.info(f"Точність: {accuracy:.2f}%")

        inference_time = self._measure_inference_time(model, device)
        logger.info(f"Середній час інференсу (одиночний): {inference_time['single']['mean']:.2f} мс")
        logger.info(f"Середній час інференсу (батч): {inference_time['batch']['mean']:.2f} мс")

        # Профілювання моделі
        profile_results = self._profile_model(model, device)

        # Збереження результатів
        result = {
            "name": model_name,
            "device": device.type,
            "size_mb": model_size,
            "parameters": num_params,
            "memory_usage_mb": memory_usage,
            "accuracy": accuracy,
            "inference_time": inference_time,
            "throughput_samples_per_sec": self.batch_size / (inference_time["batch"]["mean"] / 1000),
            "profile": profile_results,
            "timestamp": datetime.now().isoformat()
        }

        self.results[model_name] = result
        return result

    def run_all_benchmarks(self):
        '''
        Запуск бенчмаркінгу для всіх моделей

        Повертає:
        -----------
        словник з результатами бенчмаркінгу
        '''
        for model_name in self.models_dict.keys():
            self.benchmark_model(model_name)

        # Збереження результатів
        self.save_results()

        return self.results

    def save_results(self, filename=None):
        '''
        Збереження результатів бенчмаркінгу

        Параметри:
        -----------
        filename: ім'я файлу для збереження (якщо None, генерується автоматично)
        '''
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_results_{timestamp}.json"

        file_path = os.path.join(self.save_dir, filename)

        with open(file_path, 'w') as f:
            json.dump(self.results, f, indent=2)

        logger.info(f"Результати збережено у {file_path}")

    def generate_report(self, output_format='text'):
        '''
        Генерація звіту з результатами бенчмаркінгу

        Параметри:
        -----------
        output_format: формат звіту ('text', 'html', 'markdown')

        Повертає:
        -----------
        звіт у вказаному форматі
        '''
        if not self.results:
            logger.warning("Немає результатів для генерації звіту")
            return "Немає результатів для генерації звіту"

        # Підготовка даних для таблиці
        headers = ["Модель", "Розмір (МБ)", "Параметри", "Пам'ять (МБ)", "Точність (%)", "Інференс (мс)", "Пропускна здатність (зразків/с)"]
        rows = []

        for model_name, result in self.results.items():
            rows.append([
                model_name,
                f"{result['size_mb']:.2f}",
                f"{result['parameters']:,}",
                f"{result['memory_usage_mb']:.2f}",
                f"{result['accuracy']:.2f}",
                f"{result['inference_time']['single']['mean']:.2f}",
                f"{result['throughput_samples_per_sec']:.2f}"
            ])

        # Генерація таблиці у вказаному форматі
        if output_format == 'html':
            table = tabulate(rows, headers, tablefmt="html")
            report = f"<h1>Звіт з бенчмаркінгу моделей</h1>\n<p>Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>\n{table}"
        elif output_format == 'markdown':
            table = tabulate(rows, headers, tablefmt="pipe")
            report = f"# Звіт з бенчмаркінгу моделей\nДата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n{table}"
        else:  # text
            table = tabulate(rows, headers, tablefmt="grid")
            report = f"Звіт з бенчмаркінгу моделей\nДата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n{table}"

        return report

    def generate_plots(self):
        '''
        Генерація графіків з результатами бенчмаркінгу
        '''
        if not self.results:
            logger.warning("Немає результатів для генерації графіків")
            return

        # Підготовка даних для графіків
        model_names = list(self.results.keys())
        sizes = [self.results[m]['size_mb'] for m in model_names]
        accuracies = [self.results[m]['accuracy'] for m in model_names]
        inference_times = [self.results[m]['inference_time']['single']['mean'] for m in model_names]
        throughputs = [self.results[m]['throughput_samples_per_sec'] for m in model_names]

        # Створення графіків
        plt.figure(figsize=(15, 10))

        # Графік розмірів моделей
        plt.subplot(2, 2, 1)
        plt.bar(model_names, sizes, color='skyblue')
        plt.title('Розмір моделі (МБ)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        # Графік точності
        plt.subplot(2, 2, 2)
        plt.bar(model_names, accuracies, color='lightgreen')
        plt.title('Точність (%)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        # Графік часу інференсу
        plt.subplot(2, 2, 3)
        plt.bar(model_names, inference_times, color='salmon')
        plt.title('Час інференсу (мс)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        # Графік пропускної здатності
        plt.subplot(2, 2, 4)
        plt.bar(model_names, throughputs, color='mediumpurple')
        plt.title('Пропускна здатність (зразків/с)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        # Збереження графіків
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = os.path.join(self.save_dir, f"benchmark_plots_{timestamp}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"Графіки збережено у {plot_path}")

        # Додатковий графік: співвідношення точності та часу інференсу
        plt.figure(figsize=(10, 6))
        plt.scatter(inference_times, accuracies, s=100, alpha=0.7)

        # Додавання підписів до точок
        for i, model_name in enumerate(model_names):
            plt.annotate(model_name, (inference_times[i], accuracies[i]), 
                        textcoords="offset points", xytext=(0,10), ha='center')

        plt.xlabel('Час інференсу (мс)')
        plt.ylabel('Точність (%)')
        plt.title('Співвідношення точності та часу інференсу')
        plt.grid(True, linestyle='--', alpha=0.7)

        # Збереження графіку
        scatter_path = os.path.join(self.save_dir, f"accuracy_vs_speed_{timestamp}.png")
        plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
        logger.info(f"Графік співвідношення збережено у {scatter_path}")

def load_models(model_paths):
    '''
    Завантаження моделей з файлів

    Параметри:
    -----------
    model_paths: словник {"назва": "шлях"}

    Повертає:
    -----------
    словник з моделями {"назва": модель}
    '''
    models_dict = {}

    for name, path in model_paths.items():
        try:
            logger.info(f"Завантаження моделі {name} з {path}")
            model = torch.load(path, map_location=torch.device('cpu'))
            model.eval()
            models_dict[name] = model
        except Exception as e:
            logger.error(f"Помилка завантаження моделі {name}: {e}")

    return models_dict

def main():
    parser = argparse.ArgumentParser(description="Бенчмаркінг моделей машинного навчання")
    parser.add_argument("--models", type=str, required=True,
                        help="Шляхи до моделей у форматі 'назва:шлях,назва2:шлях2'")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Шлях до тестового датасету")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Розмір батчу для інференсу")
    parser.add_argument("--num-runs", type=int, default=100,
                        help="Кількість запусків для вимірювання часу інференсу")
    parser.add_argument("--save-dir", type=str, default="benchmark_results",
                        help="Директорія для збереження результатів")
    parser.add_argument("--report-format", type=str, choices=['text', 'html', 'markdown'], default="markdown",
                        help="Формат звіту")

    args = parser.parse_args()

    # Парсинг шляхів до моделей
    model_paths = {}
    for model_spec in args.models.split(','):
        if ':' in model_spec:
            name, path = model_spec.split(':', 1)
            model_paths[name] = path

    if not model_paths:
        logger.error("Не вказано жодної моделі для бенчмаркінгу")
        return

    # Завантаження моделей
    models_dict = load_models(model_paths)

    if not models_dict:
        logger.error("Не вдалося завантажити жодної моделі")
        return

    # Створення та запуск бенчмаркера
    benchmarker = ModelBenchmark(
        models_dict=models_dict,
        dataset_path=args.dataset,
        batch_size=args.batch_size,
        num_runs=args.num_runs,
        save_dir=args.save_dir
    )

    # Запуск бенчмаркінгу
    results = benchmarker.run_all_benchmarks()

    # Генерація звіту
    report = benchmarker.generate_report(output_format=args.report_format)
    report_path = os.path.join(args.save_dir, f"benchmark_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{args.report_format}")

    with open(report_path, 'w') as f:
        f.write(report)

    logger.info(f"Звіт збережено у {report_path}")

    # Генерація графіків
    benchmarker.generate_plots()

    logger.info("Бенчмаркінг завершено")

if __name__ == "__main__":
    main()
