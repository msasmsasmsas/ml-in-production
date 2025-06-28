#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Модуль з метриками для виявлення дрейфу даних
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.spatial.distance import jensenshannon
from typing import Dict, Tuple, Any, List, Union

def calculate_statistical_metrics(data: pd.Series) -> Dict[str, float]:
    """
    Обчислення статистичних метрик для числової серії даних

    Args:
        data: Серія даних для аналізу

    Returns:
        Словник зі статистичними метриками
    """
    metrics = {
        "mean": float(data.mean()),
        "std": float(data.std()),
        "min": float(data.min()),
        "max": float(data.max()),
        "median": float(data.median()),
        "q25": float(data.quantile(0.25)),
        "q75": float(data.quantile(0.75)),
        "skewness": float(stats.skew(data.dropna())),
        "kurtosis": float(stats.kurtosis(data.dropna()))
    }
    return metrics

def calculate_distribution_metrics(data: pd.Series) -> Dict[str, Any]:
    """
    Обчислення метрик розподілу для категоріальної серії даних

    Args:
        data: Серія даних для аналізу

    Returns:
        Словник з метриками розподілу
    """
    # Обчислення розподілу значень
    value_counts = data.value_counts(normalize=True)
    distribution = value_counts.to_dict()

    # Обчислення ентропії розподілу
    entropy = stats.entropy(value_counts.values)

    metrics = {
        "distribution": distribution,
        "entropy": float(entropy),
        "unique_count": len(distribution),
        "most_common": value_counts.index[0] if not value_counts.empty else None,
        "most_common_freq": float(value_counts.iloc[0]) if not value_counts.empty else 0
    }
    return metrics

def kolmogorov_smirnov_test(
    reference: np.ndarray,
    current: np.ndarray
) -> Tuple[float, float]:
    """
    Виконання тесту Колмогорова-Смирнова для порівняння двох розподілів

    Args:
        reference: Еталонний набір даних
        current: Поточний набір даних

    Returns:
        Tuple з статистикою тесту та p-value
    """
    try:
        # Видалення NaN значень
        reference_clean = reference[~np.isnan(reference)]
        current_clean = current[~np.isnan(current)]

        # Виконання тесту
        ks_stat, p_value = stats.ks_2samp(reference_clean, current_clean)
        return float(ks_stat), float(p_value)
    except Exception as e:
        # В разі помилки повертаємо нульові значення
        return 0.0, 1.0

def wasserstein_distance(reference: np.ndarray, current: np.ndarray) -> float:
    """
    Обчислення відстані Вассерштейна (Earth Mover's Distance) між двома розподілами

    Args:
        reference: Еталонний набір даних
        current: Поточний набір даних

    Returns:
        Відстань Вассерштейна
    """
    try:
        # Видалення NaN значень
        reference_clean = reference[~np.isnan(reference)]
        current_clean = current[~np.isnan(current)]

        # Обчислення відстані
        distance = stats.wasserstein_distance(reference_clean, current_clean)
        return float(distance)
    except Exception as e:
        # В разі помилки повертаємо нульове значення
        return 0.0

def jensen_shannon_divergence(reference: np.ndarray, current: np.ndarray) -> float:
    """
    Обчислення дивергенції Дженсена-Шеннона між двома розподілами

    Args:
        reference: Еталонний набір даних
        current: Поточний набір даних

    Returns:
        Дивергенція Дженсена-Шеннона
    """
    try:
        # Видалення NaN значень
        reference_clean = reference[~np.isnan(reference)]
        current_clean = current[~np.isnan(current)]

        # Для обчислення JS дивергенції потрібні оцінки густини розподілу
        # Використовуємо гістограми з однаковими бінами для обох наборів даних
        min_val = min(np.min(reference_clean), np.min(current_clean))
        max_val = max(np.max(reference_clean), np.max(current_clean))

        # Якщо дані однакові, повертаємо 0
        if min_val == max_val:
            return 0.0

        # Кількість бінів для гістограми (корінь з кількості елементів, але не менше 5)
        n_bins = max(5, int(np.sqrt(min(len(reference_clean), len(current_clean)))))

        # Обчислення гістограм
        ref_hist, _ = np.histogram(reference_clean, bins=n_bins, range=(min_val, max_val), density=True)
        current_hist, _ = np.histogram(current_clean, bins=n_bins, range=(min_val, max_val), density=True)

        # Нормалізація гістограм, щоб їх сума дорівнювала 1
        ref_hist = ref_hist / np.sum(ref_hist) if np.sum(ref_hist) > 0 else ref_hist
        current_hist = current_hist / np.sum(current_hist) if np.sum(current_hist) > 0 else current_hist

        # Обчислення JS дивергенції
        js_div = jensenshannon(ref_hist, current_hist)

        # JS дивергенція може бути NaN, якщо один з розподілів має всі нулі
        if np.isnan(js_div):
            return 0.0

        return float(js_div)
    except Exception as e:
        # В разі помилки повертаємо нульове значення
        return 0.0

def chi_square_test(reference: np.ndarray, current: np.ndarray) -> Tuple[float, float]:
    """
    Виконання хі-квадрат тесту для порівняння двох категоріальних розподілів

    Args:
        reference: Еталонний набір даних (частоти)
        current: Поточний набір даних (частоти)

    Returns:
        Tuple з статистикою тесту та p-value
    """
    try:
        # Перевірка, чи є ненульові частоти
        if np.sum(reference) == 0 or np.sum(current) == 0:
            return 0.0, 1.0

        # Виконання тесту
        chi2_stat, p_value = stats.chisquare(current, reference)
        return float(chi2_stat), float(p_value)
    except Exception as e:
        # В разі помилки повертаємо нульові значення
        return 0.0, 1.0
