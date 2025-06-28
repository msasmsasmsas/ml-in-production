#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Приклад використання детектора дрейфу даних
"""

import numpy as np
import pandas as pd
import logging
import json
import os
from datetime import datetime

from drift_detection import DriftDetector, DriftType, DriftSeverity
from visualization import DriftVisualizer

# Налаштування логування
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

def generate_synthetic_data(n_samples=1000, n_features=10, shift_factor=0.0, categorical_features=None):
    """
    Генерує синтетичні дані для демонстрації детектора дрейфу

    Args:
        n_samples: Кількість зразків
        n_features: Кількість ознак
        shift_factor: Коефіцієнт зсуву для симуляції дрейфу (0.0 = немає дрейфу, 1.0 = значний дрейф)
        categorical_features: Список індексів категоріальних ознак

    Returns:
        DataFrame з синтетичними даними
    """
    if categorical_features is None:
        categorical_features = []

    # Створення числових ознак
    data = np.random.randn(n_samples, n_features) + shift_factor

    # Додавання дрейфу до деяких ознак
    if shift_factor > 0:
        # Зсув середнього значення для перших ознак
        data[:, 0] += shift_factor * 2
        # Зміна дисперсії для других ознак
        data[:, 1] *= (1 + shift_factor)
        # Зміна розподілу для третіх ознак
        data[:, 2] = np.random.exponential(scale=1 + shift_factor, size=n_samples)

    # Створення DataFrame
    df = pd.DataFrame(data, columns=[f"feature_{i}" for i in range(n_features)])

    # Перетворення деяких ознак у категоріальні
    for idx in categorical_features:
        col_name = f"feature_{idx}"
        if col_name in df.columns:
            # Створення категоріальної ознаки з 5 категоріями
            categories = ["A", "B", "C", "D", "E"]

            # Розподіл категорій
            if shift_factor == 0:
                # Рівномірний розподіл для еталонних даних
                probs = np.ones(5) / 5
            else:
                # Зміщений розподіл для даних з дрейфом
                probs = np.array([0.1, 0.2, 0.4, 0.2, 0.1]) + shift_factor * np.array([0.2, 0.1, -0.3, 0.0, 0.0])
                probs = probs / probs.sum()  # Нормалізація, щоб сума була 1

            # Генерація категорій
            df[col_name] = np.random.choice(categories, size=n_samples, p=probs)

    return df

def main():
    # Параметри для синтетичних даних
    n_samples = 1000
    n_features = 10
    categorical_features = [3, 7, 9]  # Індекси категоріальних ознак

    # Створення директорії для результатів
    output_dir = "drift_example_results"
    os.makedirs(output_dir, exist_ok=True)

    # Генерація еталонних даних (без дрейфу)
    logger.info("Генерація еталонних даних")
    reference_data = generate_synthetic_data(
        n_samples=n_samples,
        n_features=n_features,
        shift_factor=0.0,
        categorical_features=categorical_features
    )

    # Симуляція передбачень для еталонних даних
    logger.info("Симуляція передбачень для еталонних даних")
    reference_predictions = np.random.rand(n_samples)  # Ймовірності класу 1
    reference_classes = (reference_predictions > 0.5).astype(int)  # Бінарні класи

    # Рівні дрейфу для тестування
    drift_levels = [0.0, 0.1, 0.3, 0.8]

    # Налаштування візуалізатора
    visualizer = DriftVisualizer(output_dir=output_dir)

    # Числові та категоріальні колонки
    numerical_columns = [f"feature_{i}" for i in range(n_features) if i not in categorical_features]
    categorical_columns = [f"feature_{i}" for i in categorical_features]

    # Ініціалізація детектора дрейфу
    logger.info("Ініціалізація детектора дрейфу")
    detector = DriftDetector(
        reference_data=reference_data,
        reference_predictions=reference_predictions,
        categorical_columns=categorical_columns,
        numerical_columns=numerical_columns,
        drift_threshold_low=0.05,
        drift_threshold_medium=0.1,
        drift_threshold_high=0.2
    )

    # Тестування детектора на різних рівнях дрейфу
    for drift_level in drift_levels:
        logger.info(f"\nТестування дрейфу з рівнем {drift_level}")

        # Генерація даних з дрейфом
        current_data = generate_synthetic_data(
            n_samples=n_samples,
            n_features=n_features,
            shift_factor=drift_level,
            categorical_features=categorical_features
        )

        # Симуляція передбачень для поточних даних
        # Додаємо дрейф у передбачення зі збільшенням ймовірності класу 1
        current_predictions = np.random.rand(n_samples) + drift_level * 0.2
        current_predictions = np.clip(current_predictions, 0, 1)  # Обмеження в діапазоні [0, 1]
        current_classes = (current_predictions > 0.5).astype(int)

        # Виявлення дрейфу
        drift_results = detector.detect_drift(current_data, current_predictions)

        # Виведення результатів
        overall_score = drift_results["overall_drift"]["score"]
        overall_severity = drift_results["overall_drift"]["severity"]
        logger.info(f"Загальний рівень дрейфу: {overall_score:.4f} ({overall_severity})")

        # Збереження результатів
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = os.path.join(output_dir, f"drift_results_level_{drift_level}_{timestamp}.json")
        with open(result_file, 'w') as f:
            json.dump(drift_results, f, indent=2, default=str)
        logger.info(f"Результати збережено в {result_file}")

        # Створення візуалізацій
        visualizer.visualize_drift_results(drift_results, reference_data, current_data)
        logger.info(f"Візуалізації збережено в {output_dir}")

if __name__ == "__main__":
    main()
