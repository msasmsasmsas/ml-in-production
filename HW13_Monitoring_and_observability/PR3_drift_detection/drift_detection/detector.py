#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Модуль для виявлення дрейфу даних у вхідних ознаках та вихідних даних моделі
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
import logging
from typing import Dict, List, Tuple, Any, Optional, Union
from enum import Enum
from datetime import datetime

from drift_detection.metrics import (
    calculate_statistical_metrics,
    calculate_distribution_metrics,
    jensen_shannon_divergence,
    kolmogorov_smirnov_test,
    wasserstein_distance
)

logger = logging.getLogger(__name__)

class DriftType(str, Enum):
    """
    Типи дрейфу даних
    """
    DATA_DRIFT = "data_drift"  # Дрейф у вхідних даних
    PREDICTION_DRIFT = "prediction_drift"  # Дрейф у передбаченнях
    CONCEPT_DRIFT = "concept_drift"  # Дрейф у взаємозв'язку між входом і виходом

class DriftSeverity(str, Enum):
    """
    Рівні серйозності дрейфу
    """
    NO_DRIFT = "no_drift"  # Дрейфу немає
    LOW = "low"  # Низький рівень дрейфу
    MEDIUM = "medium"  # Середній рівень дрейфу
    HIGH = "high"  # Високий рівень дрейфу

class DriftDetector:
    """
    Клас для виявлення різних типів дрейфу даних
    """
    def __init__(
        self,
        reference_data: Optional[pd.DataFrame] = None,
        reference_predictions: Optional[np.ndarray] = None,
        categorical_columns: Optional[List[str]] = None,
        numerical_columns: Optional[List[str]] = None,
        drift_threshold_low: float = 0.05,
        drift_threshold_medium: float = 0.1,
        drift_threshold_high: float = 0.2
    ):
        """
        Ініціалізація детектора дрейфу

        Args:
            reference_data: Еталонний набір даних для порівняння
            reference_predictions: Еталонні передбачення моделі
            categorical_columns: Список категоріальних стовпців
            numerical_columns: Список числових стовпців
            drift_threshold_low: Поріг для низького рівня дрейфу
            drift_threshold_medium: Поріг для середнього рівня дрейфу
            drift_threshold_high: Поріг для високого рівня дрейфу
        """
        self.reference_data = reference_data
        self.reference_predictions = reference_predictions
        self.categorical_columns = categorical_columns or []
        self.numerical_columns = numerical_columns or []
        self.drift_threshold_low = drift_threshold_low
        self.drift_threshold_medium = drift_threshold_medium
        self.drift_threshold_high = drift_threshold_high

        # Зберігаємо статистики для еталонних даних
        self.reference_stats = {}
        if reference_data is not None:
            self._calculate_reference_statistics()

        logger.info("Ініціалізовано детектор дрейфу даних")

    def _calculate_reference_statistics(self):
        """
        Обчислення статистик для еталонних даних
        """
        logger.info("Обчислення статистик для еталонних даних")

        # Для числових стовпців
        for col in self.numerical_columns:
            if col in self.reference_data.columns:
                self.reference_stats[col] = calculate_statistical_metrics(self.reference_data[col])

        # Для категоріальних стовпців
        for col in self.categorical_columns:
            if col in self.reference_data.columns:
                self.reference_stats[col] = calculate_distribution_metrics(self.reference_data[col])

        # Якщо є еталонні передбачення
        if self.reference_predictions is not None:
            self.reference_stats["predictions"] = calculate_distribution_metrics(
                pd.Series(self.reference_predictions))

    def set_reference_data(self, reference_data: pd.DataFrame, reference_predictions: Optional[np.ndarray] = None):
        """
        Встановлення або оновлення еталонних даних

        Args:
            reference_data: Новий еталонний набір даних
            reference_predictions: Нові еталонні передбачення
        """
        self.reference_data = reference_data
        if reference_predictions is not None:
            self.reference_predictions = reference_predictions

        # Перерахунок статистик
        self._calculate_reference_statistics()
        logger.info("Оновлено еталонні дані та статистики")

    def detect_drift(
        self,
        current_data: pd.DataFrame,
        current_predictions: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Виявлення дрейфу в поточних даних порівняно з еталонними

        Args:
            current_data: Поточний набір даних для аналізу
            current_predictions: Поточні передбачення моделі (опціонально)

        Returns:
            Словник з результатами виявлення дрейфу
        """
        if self.reference_data is None:
            raise ValueError("Еталонні дані не встановлено. Використайте set_reference_data()")

        drift_results = {
            "timestamp": datetime.now().isoformat(),
            "data_drift": {},
            "prediction_drift": None,
            "concept_drift": None,
            "overall_drift": {
                "score": 0.0,
                "severity": DriftSeverity.NO_DRIFT
            }
        }

        # Перевірка дрейфу для кожної ознаки
        feature_drift_scores = []

        # Для числових ознак
        for col in self.numerical_columns:
            if col in current_data.columns and col in self.reference_data.columns:
                drift_score, drift_details = self._detect_numerical_drift(
                    self.reference_data[col], current_data[col])

                severity = self._get_drift_severity(drift_score)

                drift_results["data_drift"][col] = {
                    "score": drift_score,
                    "severity": severity,
                    "details": drift_details
                }

                feature_drift_scores.append(drift_score)

        # Для категоріальних ознак
        for col in self.categorical_columns:
            if col in current_data.columns and col in self.reference_data.columns:
                drift_score, drift_details = self._detect_categorical_drift(
                    self.reference_data[col], current_data[col])

                severity = self._get_drift_severity(drift_score)

                drift_results["data_drift"][col] = {
                    "score": drift_score,
                    "severity": severity,
                    "details": drift_details
                }

                feature_drift_scores.append(drift_score)

        # Перевірка дрейфу в передбаченнях, якщо вони надані
        if current_predictions is not None and self.reference_predictions is not None:
            pred_drift_score, pred_drift_details = self._detect_prediction_drift(
                self.reference_predictions, current_predictions)

            pred_severity = self._get_drift_severity(pred_drift_score)

            drift_results["prediction_drift"] = {
                "score": pred_drift_score,
                "severity": pred_severity,
                "details": pred_drift_details
            }

            feature_drift_scores.append(pred_drift_score)

        # Обчислення загального рівня дрейфу
        if feature_drift_scores:
            overall_drift_score = np.mean(feature_drift_scores)
            overall_severity = self._get_drift_severity(overall_drift_score)

            drift_results["overall_drift"] = {
                "score": float(overall_drift_score),
                "severity": overall_severity
            }

        return drift_results

    def _detect_numerical_drift(
        self,
        reference_data: pd.Series,
        current_data: pd.Series
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Виявлення дрейфу для числових ознак

        Args:
            reference_data: Еталонні дані ознаки
            current_data: Поточні дані ознаки

        Returns:
            Tuple з оцінкою дрейфу та деталями
        """
        details = {}

        # Обчислення статистичних метрик для поточних даних
        current_stats = calculate_statistical_metrics(current_data)
        ref_stats = self.reference_stats.get(reference_data.name, {})

        # Порівняння статистик
        if ref_stats:
            for metric, value in current_stats.items():
                if metric in ref_stats:
                    # Обчислення відносної зміни
                    if ref_stats[metric] != 0:
                        rel_change = abs(value - ref_stats[metric]) / abs(ref_stats[metric])
                    else:
                        rel_change = 1.0 if value != 0 else 0.0

                    details[f"{metric}_change"] = rel_change

        # Виконання статистичних тестів
        # 1. Тест Колмогорова-Смирнова
        ks_stat, ks_pvalue = kolmogorov_smirnov_test(reference_data.values, current_data.values)
        details["ks_statistic"] = ks_stat
        details["ks_pvalue"] = ks_pvalue

        # 2. Відстань Вассерштейна
        wd = wasserstein_distance(reference_data.values, current_data.values)
        details["wasserstein_distance"] = wd

        # 3. Дивергенція Дженсена-Шеннона (для дискретизованих даних)
        js_div = jensen_shannon_divergence(reference_data.values, current_data.values)
        details["jensen_shannon_divergence"] = js_div

        # Обчислення загальної оцінки дрейфу (на основі p-value та метрик відстані)
        # Перетворення p-value в оцінку дрейфу (менше p-value -> більший дрейф)
        pvalue_score = 1.0 - min(ks_pvalue, 1.0)

        # Нормалізація відстані Вассерштейна (залежить від масштабу даних)
        ref_range = ref_stats.get("max", 0) - ref_stats.get("min", 0)
        if ref_range > 0:
            wd_score = min(wd / ref_range, 1.0)
        else:
            wd_score = 0.0

        # Об'єднання метрик в одну оцінку
        drift_score = 0.5 * pvalue_score + 0.3 * wd_score + 0.2 * js_div

        return drift_score, details

    def _detect_categorical_drift(
        self,
        reference_data: pd.Series,
        current_data: pd.Series
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Виявлення дрейфу для категоріальних ознак

        Args:
            reference_data: Еталонні дані ознаки
            current_data: Поточні дані ознаки

        Returns:
            Tuple з оцінкою дрейфу та деталями
        """
        details = {}

        # Обчислення розподілів категорій
        ref_dist = reference_data.value_counts(normalize=True).to_dict()
        current_dist = current_data.value_counts(normalize=True).to_dict()

        # Об'єднання всіх категорій з обох наборів даних
        all_categories = set(ref_dist.keys()) | set(current_dist.keys())

        # Заповнення відсутніх категорій нулями
        ref_dist_complete = {cat: ref_dist.get(cat, 0) for cat in all_categories}
        current_dist_complete = {cat: current_dist.get(cat, 0) for cat in all_categories}

        # Перетворення в масиви для обчислення метрик
        ref_array = np.array(list(ref_dist_complete.values()))
        current_array = np.array(list(current_dist_complete.values()))

        # Обчислення дивергенції Дженсена-Шеннона
        js_div = jensen_shannon_divergence(ref_array, current_array)
        details["jensen_shannon_divergence"] = js_div

        # Обчислення хі-квадрат тесту для порівняння розподілів
        # Потрібно перетворити відсотки у кількості
        ref_size = len(reference_data)
        current_size = len(current_data)

        # Перевірка, чи достатньо даних для тесту хі-квадрат
        if ref_size >= 5 and current_size >= 5 and len(all_categories) > 1:
            try:
                ref_counts = np.array([ref_dist_complete[cat] * ref_size for cat in all_categories])
                current_counts = np.array([current_dist_complete[cat] * current_size for cat in all_categories])

                # Запобігаємо помилкам, якщо є нульові очікувані частоти
                mask = ref_counts > 0
                if sum(mask) > 1:  # Потрібно принаймні дві ненульові категорії
                    chi2_stat, chi2_pvalue = stats.chisquare(current_counts[mask], ref_counts[mask])
                    details["chi2_statistic"] = chi2_stat
                    details["chi2_pvalue"] = chi2_pvalue

                    # Перетворення p-value в оцінку дрейфу
                    chi2_score = 1.0 - min(chi2_pvalue, 1.0)
                else:
                    chi2_score = 0.0
            except Exception as e:
                logger.warning(f"Помилка при обчисленні хі-квадрат тесту: {str(e)}")
                chi2_score = 0.0
        else:
            chi2_score = 0.0

        # Обчислення різниці в розподілах для кожної категорії
        category_changes = {}
        for cat in all_categories:
            ref_prob = ref_dist_complete.get(cat, 0)
            current_prob = current_dist_complete.get(cat, 0)
            abs_change = abs(current_prob - ref_prob)
            if ref_prob > 0:
                rel_change = abs_change / ref_prob
            else:
                rel_change = 1.0 if current_prob > 0 else 0.0

            category_changes[str(cat)] = {
                "reference_prob": ref_prob,
                "current_prob": current_prob,
                "absolute_change": abs_change,
                "relative_change": rel_change
            }

        details["category_changes"] = category_changes

        # Обчислення максимальної відносної зміни по категоріях
        max_rel_change = max([info["relative_change"] for info in category_changes.values()], default=0)
        details["max_relative_change"] = max_rel_change

        # Обчислення загальної оцінки дрейфу для категоріальних даних
        # Комбінуємо JS дивергенцію, хі-квадрат оцінку та максимальну відносну зміну
        drift_score = 0.4 * js_div + 0.4 * chi2_score + 0.2 * min(max_rel_change, 1.0)

        return drift_score, details

    def _detect_prediction_drift(
        self,
        reference_predictions: np.ndarray,
        current_predictions: np.ndarray
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Виявлення дрейфу у передбаченнях моделі

        Args:
            reference_predictions: Еталонні передбачення
            current_predictions: Поточні передбачення

        Returns:
            Tuple з оцінкою дрейфу та деталями
        """
        details = {}

        # Перетворення в одновимірні масиви, якщо потрібно
        ref_preds = reference_predictions.flatten() if reference_predictions.ndim > 1 else reference_predictions
        current_preds = current_predictions.flatten() if current_predictions.ndim > 1 else current_predictions

        # Обчислення статистичних метрик
        ref_stats = calculate_statistical_metrics(pd.Series(ref_preds))
        current_stats = calculate_statistical_metrics(pd.Series(current_preds))

        # Порівняння статистик
        stat_changes = {}
        for metric, value in current_stats.items():
            if metric in ref_stats:
                # Обчислення відносної зміни
                if ref_stats[metric] != 0:
                    rel_change = abs(value - ref_stats[metric]) / abs(ref_stats[metric])
                else:
                    rel_change = 1.0 if value != 0 else 0.0

                stat_changes[f"{metric}_change"] = rel_change

        details["statistic_changes"] = stat_changes

        # Тест Колмогорова-Смирнова
        ks_stat, ks_pvalue = kolmogorov_smirnov_test(ref_preds, current_preds)
        details["ks_statistic"] = ks_stat
        details["ks_pvalue"] = ks_pvalue

        # Відстань Вассерштейна
        wd = wasserstein_distance(ref_preds, current_preds)
        details["wasserstein_distance"] = wd

        # Дивергенція Дженсена-Шеннона
        js_div = jensen_shannon_divergence(ref_preds, current_preds)
        details["jensen_shannon_divergence"] = js_div

        # Обчислення загальної оцінки дрейфу (комбінація різних метрик)
        # Перетворення p-value в оцінку дрейфу
        pvalue_score = 1.0 - min(ks_pvalue, 1.0)

        # Нормалізація відстані Вассерштейна
        ref_range = ref_stats.get("max", 0) - ref_stats.get("min", 0)
        if ref_range > 0:
            wd_score = min(wd / ref_range, 1.0)
        else:
            wd_score = 0.0

        # Середня відносна зміна ключових статистик
        key_stats = ["mean_change", "std_change", "median_change"]
        mean_stat_change = np.mean([stat_changes.get(k, 0) for k in key_stats])

        # Об'єднання метрик в одну оцінку
        drift_score = 0.3 * pvalue_score + 0.2 * wd_score + 0.3 * js_div + 0.2 * min(mean_stat_change, 1.0)

        return drift_score, details

    def _get_drift_severity(self, drift_score: float) -> DriftSeverity:
        """
        Визначення рівня серйозності дрейфу на основі оцінки

        Args:
            drift_score: Оцінка дрейфу (0-1)

        Returns:
            Рівень серйозності дрейфу
        """
        if drift_score < self.drift_threshold_low:
            return DriftSeverity.NO_DRIFT
        elif drift_score < self.drift_threshold_medium:
            return DriftSeverity.LOW
        elif drift_score < self.drift_threshold_high:
            return DriftSeverity.MEDIUM
        else:
            return DriftSeverity.HIGH
