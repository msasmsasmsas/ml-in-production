#!/usr/bin/env python
"""
Інструмент для бенчмаркінгу окремих компонентів сервера моделей машинного навчання
"""

import os
import time
import json
import argparse
import statistics
import concurrent.futures
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision.models import resnet50, ResNet50_Weights

class ComponentBenchmarker:
    """
    Клас для бенчмаркінгу окремих компонентів інференсу моделі
    """
    def __init__(self, model_name='resnet50', device=None):
        """
        Ініціалізація бенчмаркера

        Параметри:
        -----------
        model_name: назва моделі
        device: пристрій для виконання (cuda або cpu)
        """
        self.model_name = model_name

        # Визначення пристрою
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        print(f"Використання пристрою: {self.device}")

        # Завантаження моделі
        self.model = self._load_model()

        # Створення трансформацій для зображень
        self.preprocessing = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def _load_model(self):
        """
        Завантаження моделі

        Повертає:
        -----------
        модель PyTorch
        """
        if self.model_name == 'resnet50':
            model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        else:
            raise ValueError(f"Непідтримувана модель: {self.model_name}")

        model.to(self.device)
        model.eval()
        return model

    def benchmark_image_loading(self, image_path, iterations=100):
        """
        Бенчмаркінг завантаження зображень

        Параметри:
        -----------
        image_path: шлях до файлу зображення
        iterations: кількість ітерацій

        Повертає:
        -----------
        словник з результатами
        """
        results = []

        for _ in range(iterations):
            start_time = time.time()
            with Image.open(image_path) as img:
                img_copy = img.copy()
            elapsed = time.time() - start_time
            results.append(elapsed)

        return self._calculate_stats(results, 'image_loading')

    def benchmark_preprocessing(self, image_path, iterations=100):
        """
        Бенчмаркінг попередньої обробки зображень

        Параметри:
        -----------
        image_path: шлях до файлу зображення
        iterations: кількість ітерацій

        Повертає:
        -----------
        словник з результатами
        """
        with Image.open(image_path) as img:
            img_copy = img.copy()

        results = []

        for _ in range(iterations):
            start_time = time.time()
            tensor = self.preprocessing(img_copy)
            elapsed = time.time() - start_time
            results.append(elapsed)

        return self._calculate_stats(results, 'preprocessing')

    def benchmark_model_forward(self, image_path, batch_size=1, iterations=100):
        """
        Бенчмаркінг проходу вперед моделі

        Параметри:
        -----------
        image_path: шлях до файлу зображення
        batch_size: розмір батчу
        iterations: кількість ітерацій

        Повертає:
        -----------
        словник з результатами
        """
        # Підготовка вхідних даних
        with Image.open(image_path) as img:
            tensor = self.preprocessing(img).unsqueeze(0).to(self.device)

        # Створення батчу
        if batch_size > 1:
            tensor = tensor.repeat(batch_size, 1, 1, 1)

        results = []

        # Прогрів GPU
        if self.device.type == 'cuda':
            print("Прогрів GPU...")
            for _ in range(10):
                with torch.no_grad():
                    self.model(tensor)
            torch.cuda.synchronize()

        for _ in range(iterations):
            start_time = time.time()

            with torch.no_grad():
                outputs = self.model(tensor)

            if self.device.type == 'cuda':
                torch.cuda.synchronize()

            elapsed = time.time() - start_time
            results.append(elapsed)

        return self._calculate_stats(results, f'model_forward_batch{batch_size}')

    def benchmark_postprocessing(self, image_path, iterations=100):
        """
        Бенчмаркінг післяобробки результатів

        Параметри:
        -----------
        image_path: шлях до файлу зображення
        iterations: кількість ітерацій

        Повертає:
        -----------
        словник з результатами
        """
        # Підготовка вхідних даних
        with Image.open(image_path) as img:
            tensor = self.preprocessing(img).unsqueeze(0).to(self.device)

        # Отримання прогнозу
        with torch.no_grad():
            outputs = self.model(tensor)

        results = []

        for _ in range(iterations):
            start_time = time.time()

            # Типова післяобробка для класифікації
            probs = torch.nn.functional.softmax(outputs, dim=1)[0]
            top5_probs, top5_indices = torch.topk(probs, 5)

            # Конвертація у numpy та форматування результатів
            top5_probs = top5_probs.cpu().numpy()
            top5_indices = top5_indices.cpu().numpy()

            predictions = []
            for i, (idx, prob) in enumerate(zip(top5_indices, top5_probs)):
                predictions.append({'class_id': int(idx), 'score': float(prob)})

            elapsed = time.time() - start_time
            results.append(elapsed)

        return self._calculate_stats(results, 'postprocessing')

    def benchmark_end_to_end(self, image_path, iterations=100):
        """
        Бенчмаркінг повного процесу інференсу

        Параметри:
        -----------
        image_path: шлях до файлу зображення
        iterations: кількість ітерацій

        Повертає:
        -----------
        словник з результатами
        """
        results = []

        for _ in range(iterations):
            start_time = time.time()

            # Завантаження зображення
            with Image.open(image_path) as img:
                img_copy = img.copy()

            # Попередня обробка
            tensor = self.preprocessing(img_copy).unsqueeze(0).to(self.device)

            # Прогнозування
            with torch.no_grad():
                outputs = self.model(tensor)

            if self.device.type == 'cuda':
                torch.cuda.synchronize()

            # Післяобробка
            probs = torch.nn.functional.softmax(outputs, dim=1)[0]
            top5_probs, top5_indices = torch.topk(probs, 5)

            top5_probs = top5_probs.cpu().numpy()
            top5_indices = top5_indices.cpu().numpy()

            predictions = []
            for i, (idx, prob) in enumerate(zip(top5_indices, top5_probs)):
                predictions.append({'class_id': int(idx), 'score': float(prob)})

            elapsed = time.time() - start_time
            results.append(elapsed)

        return self._calculate_stats(results, 'end_to_end')

    def benchmark_batch_sizes(self, image_path, batch_sizes=[1, 2, 4, 8, 16, 32], iterations=50):
        """
        Бенчмаркінг різних розмірів батчів

        Параметри:
        -----------
        image_path: шлях до файлу зображення
        batch_sizes: список розмірів батчів для тестування
        iterations: кількість ітерацій для кожного розміру батчу

        Повертає:
        -----------
        список словників з результатами
        """
        results = []

        for batch_size in batch_sizes:
            print(f"Тестування розміру батчу {batch_size}...")
            result = self.benchmark_model_forward(image_path, batch_size=batch_size, iterations=iterations)
            results.append(result)

        return results

    def run_all_benchmarks(self, image_path, iterations=100):
        """
        Запуск всіх бенчмарків

        Параметри:
        -----------
        image_path: шлях до файлу зображення
        iterations: кількість ітерацій

        Повертає:
        -----------
        словник з результатами всіх компонентів
        """
        results = {}

        print("Бенчмаркінг завантаження зображень...")
        results['image_loading'] = self.benchmark_image_loading(image_path, iterations)

        print("Бенчмаркінг попередньої обробки...")
        results['preprocessing'] = self.benchmark_preprocessing(image_path, iterations)

        print("Бенчмаркінг проходу вперед моделі...")
        results['model_forward'] = self.benchmark_model_forward(image_path, iterations=iterations)

        print("Бенчмаркінг післяобробки...")
        results['postprocessing'] = self.benchmark_postprocessing(image_path, iterations)

        print("Бенчмаркінг повного процесу інференсу...")
        results['end_to_end'] = self.benchmark_end_to_end(image_path, iterations)

        return results

    def _calculate_stats(self, times, name):
        """
        Обчислення статистики часу виконання

        Параметри:
        -----------
        times: список часів виконання
        name: назва компонента

        Повертає:
        -----------
        словник зі статистикою
        """
        # Конвертація у мілісекунди
        times_ms = [t * 1000 for t in times]

        return {
            'name': name,
            'iterations': len(times_ms),
            'min_ms': min(times_ms),
            'max_ms': max(times_ms),
            'mean_ms': statistics.mean(times_ms),
            'median_ms': statistics.median(times_ms),
            'p90_ms': np.percentile(times_ms, 90),
            'p95_ms': np.percentile(times_ms, 95),
            'p99_ms': np.percentile(times_ms, 99),
            'std_ms': statistics.stdev(times_ms) if len(times_ms) > 1 else 0,
            'raw_times_ms': times_ms
        }

def print_stats(stats):
    """
    Виведення статистики

    Параметри:
    -----------
    stats: словник зі статистикою
    """
    print(f"\nСтатистика для {stats['name']} ({stats['iterations']} ітерацій):")
    print(f"  Мін: {stats['min_ms']:.3f} мс")
    print(f"  Макс: {stats['max_ms']:.3f} мс")
    print(f"  Середнє: {stats['mean_ms']:.3f} мс")
    print(f"  Медіана: {stats['median_ms']:.3f} мс")
    print(f"  P90: {stats['p90_ms']:.3f} мс")
    print(f"  P95: {stats['p95_ms']:.3f} мс")
    print(f"  P99: {stats['p99_ms']:.3f} мс")
    print(f"  Стандартне відхилення: {stats['std_ms']:.3f} мс")

def save_results_json(results, output_file):
    """
    Зберігає результати у JSON файл

    Параметри:
    -----------
    results: словник з результатами
    output_file: шлях до вихідного файлу
    """
    # Створення копії результатів без raw_times для компактності
    results_copy = {}

    for component, stats in results.items():
        if isinstance(stats, list):  # для результатів batch_sizes
            results_copy[component] = []
            for stat in stats:
                stat_copy = stat.copy()
                if 'raw_times_ms' in stat_copy:
                    del stat_copy['raw_times_ms']
                results_copy[component].append(stat_copy)
        else:
            stats_copy = stats.copy()
            if 'raw_times_ms' in stats_copy:
                del stats_copy['raw_times_ms']
            results_copy[component] = stats_copy

    with open(output_file, 'w') as f:
        json.dump(results_copy, f, indent=2)

    print(f"Результати збережено у {output_file}")

def plot_component_comparison(results, output_file=None):
    """
    Створює графіки порівняння компонентів

    Параметри:
    -----------
    results: словник з результатами
    output_file: шлях до вихідного файлу (якщо None, графіки відображаються)
    """
    components = ['image_loading', 'preprocessing', 'model_forward', 'postprocessing', 'end_to_end']
    available_components = [c for c in components if c in results]

    if not available_components:
        print("Недостатньо даних для побудови графіків")
        return

    # Створення графіків
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Порівняння компонентів інференсу', fontsize=16)

    # Графік 1: Час виконання компонентів
    names = [results[c]['name'] for c in available_components]
    means = [results[c]['mean_ms'] for c in available_components]
    p95s = [results[c]['p95_ms'] for c in available_components]

    # Створення двох графіків на одному полотні
    ax1.bar(names, means, label='Середній час')
    ax1.bar(names, p95s, alpha=0.5, label='P95')
    ax1.set_title('Час виконання компонентів')
    ax1.set_ylabel('Час (мс)')
    ax1.set_yscale('log')  # логарифмічна шкала для кращої візуалізації
    ax1.legend()
    ax1.grid(axis='y', linestyle='--', alpha=0.7)

    # Графік 2: Відносний вклад компонентів
    # Для цього графіка використовуємо тільки середні значення
    total_time = results['end_to_end']['mean_ms']
    other_components_time = sum([results[c]['mean_ms'] for c in available_components if c != 'end_to_end'])
    overhead_time = max(0, total_time - other_components_time)

    components_for_pie = available_components.copy()
    if 'end_to_end' in components_for_pie:
        components_for_pie.remove('end_to_end')

    labels = [results[c]['name'] for c in components_for_pie]
    sizes = [results[c]['mean_ms'] for c in components_for_pie]

    if overhead_time > 0:
        labels.append('overhead')
        sizes.append(overhead_time)

    ax2.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, shadow=True)
    ax2.axis('equal')  # рівні пропорції для кругової діаграми
    ax2.set_title('Відносний вклад компонентів')

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if output_file:
        plt.savefig(output_file)
        print(f"Графік збережено у {output_file}")
    else:
        plt.show()

def plot_batch_size_comparison(batch_results, output_file=None):
    """
    Створює графіки порівняння різних розмірів батчів

    Параметри:
    -----------
    batch_results: список результатів для різних розмірів батчів
    output_file: шлях до вихідного файлу (якщо None, графіки відображаються)
    """
    if not batch_results:
        print("Недостатньо даних для побудови графіків")
        return

    # Отримання розмірів батчів з назв
    batch_sizes = []
    mean_times = []
    throughputs = []  # зображень на секунду

    for result in batch_results:
        # Очікуємо, що назва має формат 'model_forward_batchX'
        batch_size = int(result['name'].split('batch')[1])
        batch_sizes.append(batch_size)
        mean_times.append(result['mean_ms'])
        throughputs.append(batch_size * 1000 / result['mean_ms'])  # зображень на секунду

    # Створення графіків
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Порівняння розмірів батчів', fontsize=16)

    # Графік 1: Час виконання батчу
    ax1.plot(batch_sizes, mean_times, 'o-', label='Середній час')
    ax1.set_title('Час виконання батчу')
    ax1.set_xlabel('Розмір батчу')
    ax1.set_ylabel('Час (мс)')
    ax1.grid(True)

    # Графік 2: Пропускна здатність
    ax2.plot(batch_sizes, throughputs, 'o-', label='Пропускна здатність')
    ax2.set_title('Пропускна здатність')
    ax2.set_xlabel('Розмір батчу')
    ax2.set_ylabel('Зображень на секунду')
    ax2.grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if output_file:
        plt.savefig(output_file)
        print(f"Графік збережено у {output_file}")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='Інструмент для бенчмаркінгу компонентів інференсу моделі')

    # Основні параметри
    parser.add_argument('--image', type=str, required=True,
                        help='Шлях до тестового зображення')
    parser.add_argument('--model', type=str, default='resnet50',
                        help='Назва моделі')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], default=None,
                        help='Пристрій для виконання (cuda або cpu)')

    # Параметри бенчмаркінгу
    parser.add_argument('--iterations', type=int, default=100,
                        help='Кількість ітерацій для кожного тесту')
    parser.add_argument('--component', type=str, default='all',
                        choices=['all', 'image_loading', 'preprocessing', 'model_forward', 'postprocessing', 'end_to_end', 'batch_sizes'],
                        help='Компонент для бенчмаркінгу')
    parser.add_argument('--batch-sizes', type=str, default='1,2,4,8,16,32',
                        help='Розміри батчів для тестування (через кому)')

    # Параметри вихідних даних
    parser.add_argument('--output-json', type=str, default=None,
                        help='Шлях для збереження результатів у JSON')
    parser.add_argument('--output-plot', type=str, default=None,
                        help='Шлях для збереження графіків')

    args = parser.parse_args()

    # Перевірка наявності файлу зображення
    if not os.path.isfile(args.image):
        print(f"Помилка: файл {args.image} не існує")
        return 1

    # Створення бенчмаркера
    benchmarker = ComponentBenchmarker(model_name=args.model, device=args.device)

    # Запуск бенчмаркінгу
    results = {}

    if args.component == 'all':
        results = benchmarker.run_all_benchmarks(args.image, args.iterations)

        # Виведення результатів
        for component, stats in results.items():
            print_stats(stats)

        # Побудова графіків
        if args.output_plot:
            output_file = args.output_plot
        else:
            output_file = None

        plot_component_comparison(results, output_file)

    elif args.component == 'batch_sizes':
        # Парсинг розмірів батчів
        batch_sizes = list(map(int, args.batch_sizes.split(',')))

        # Запуск бенчмаркінгу
        batch_results = benchmarker.benchmark_batch_sizes(args.image, batch_sizes, args.iterations)
        results['batch_sizes'] = batch_results

        # Виведення результатів
        for stats in batch_results:
            print_stats(stats)

        # Побудова графіків
        if args.output_plot:
            output_file = args.output_plot
        else:
            output_file = None

        plot_batch_size_comparison(batch_results, output_file)

    else:
        # Запуск конкретного компонента
        benchmark_method = getattr(benchmarker, f"benchmark_{args.component}")
        component_results = benchmark_method(args.image, args.iterations)
        results[args.component] = component_results

        # Виведення результатів
        print_stats(component_results)

    # Збереження результатів
    if args.output_json:
        save_results_json(results, args.output_json)

    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
