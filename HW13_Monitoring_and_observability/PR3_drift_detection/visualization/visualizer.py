#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Модуль для візуалізації результатів виявлення дрейфу даних
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from typing import Dict, List, Any, Tuple, Optional
import os
from datetime import datetime

from drift_detection import DriftSeverity

class DriftVisualizer:
    """
    Клас для візуалізації результатів виявлення дрейфу даних
    """
    def __init__(self, output_dir: str = "drift_visualizations"):
        """
        Ініціалізація візуалізатора

        Args:
            output_dir: Директорія для збереження візуалізацій
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Налаштування стилю графіків
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_context("talk")

    def visualize_drift_results(self, drift_results: Dict[str, Any], reference_data: Optional[pd.DataFrame] = None, current_data: Optional[pd.DataFrame] = None):
        """
        Створення візуалізацій на основі результатів виявлення дрейфу

        Args:
            drift_results: Результати виявлення дрейфу
            reference_data: Еталонний набір даних (опціонально)
            current_data: Поточний набір даних (опціонально)
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Візуалізація загального рівня дрейфу
        self._plot_overall_drift(drift_results, timestamp)

        # Візуалізація дрейфу для окремих ознак
        if "data_drift" in drift_results and drift_results["data_drift"]:
            self._plot_feature_drift(drift_results["data_drift"], timestamp)

            # Якщо доступні дані, створюємо детальні візуалізації для ознак
            if reference_data is not None and current_data is not None:
                self._plot_detailed_distributions(
                    drift_results["data_drift"],
                    reference_data,
                    current_data,
                    timestamp
                )

        # Візуалізація дрейфу передбачень, якщо він є
        if "prediction_drift" in drift_results and drift_results["prediction_drift"] is not None:
            self._plot_prediction_drift(drift_results["prediction_drift"], timestamp)

    def _plot_overall_drift(self, drift_results: Dict[str, Any], timestamp: str):
        """
        Візуалізація загального рівня дрейфу

        Args:
            drift_results: Результати виявлення дрейфу
            timestamp: Часова мітка для назви файлу
        """
        overall_drift = drift_results.get("overall_drift", {})
        if not overall_drift:
            return

        score = overall_drift.get("score", 0)
        severity = overall_drift.get("severity", DriftSeverity.NO_DRIFT)

        # Створюємо діаграму з рівнем дрейфу
        fig, ax = plt.subplots(figsize=(10, 6))

        # Визначаємо колір на основі рівня серйозності
        if severity == DriftSeverity.HIGH:
            color = "red"
        elif severity == DriftSeverity.MEDIUM:
            color = "orange"
        elif severity == DriftSeverity.LOW:
            color = "yellow"
        else:
            color = "green"

        # Горизонтальна шкала від 0 до 1
        ax.barh(["Загальний дрейф"], [score], color=color)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Рівень дрейфу")
        ax.set_title(f"Загальний рівень дрейфу: {score:.4f} ({severity})")

        # Додаємо вертикальні лінії для порогів серйозності
        ax.axvline(x=0.05, color='green', linestyle='--', alpha=0.7, label="Низький")
        ax.axvline(x=0.1, color='yellow', linestyle='--', alpha=0.7, label="Середній")
        ax.axvline(x=0.2, color='red', linestyle='--', alpha=0.7, label="Високий")
        ax.legend()

        # Зберігаємо діаграму
        output_file = os.path.join(self.output_dir, f"overall_drift_{timestamp}.png")
        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()

    def _plot_feature_drift(self, data_drift: Dict[str, Any], timestamp: str):
        """
        Візуалізація дрейфу для окремих ознак

        Args:
            data_drift: Результати виявлення дрейфу для ознак
            timestamp: Часова мітка для назви файлу
        """
        if not data_drift:
            return

        # Створюємо DataFrame з результатами для кожної ознаки
        features = []
        scores = []
        severities = []
        colors = []

        for feature, info in data_drift.items():
            features.append(feature)
            scores.append(info.get("score", 0))
            severity = info.get("severity", DriftSeverity.NO_DRIFT)
            severities.append(severity)

            # Визначаємо колір на основі рівня серйозності
            if severity == DriftSeverity.HIGH:
                colors.append("red")
            elif severity == DriftSeverity.MEDIUM:
                colors.append("orange")
            elif severity == DriftSeverity.LOW:
                colors.append("yellow")
            else:
                colors.append("green")

        # Сортуємо дані за рівнем дрейфу
        sorted_indices = np.argsort(scores)
        features = [features[i] for i in sorted_indices]
        scores = [scores[i] for i in sorted_indices]
        severities = [severities[i] for i in sorted_indices]
        colors = [colors[i] for i in sorted_indices]

        # Створюємо горизонтальну гістограму
        fig, ax = plt.subplots(figsize=(10, max(6, len(features) * 0.5)))

        y_pos = np.arange(len(features))
        ax.barh(y_pos, scores, color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Рівень дрейфу")
        ax.set_title("Дрейф для окремих ознак")

        # Додаємо вертикальні лінії для порогів серйозності
        ax.axvline(x=0.05, color='green', linestyle='--', alpha=0.7, label="Низький")
        ax.axvline(x=0.1, color='yellow', linestyle='--', alpha=0.7, label="Середній")
        ax.axvline(x=0.2, color='red', linestyle='--', alpha=0.7, label="Високий")
        ax.legend()

        # Зберігаємо діаграму
        output_file = os.path.join(self.output_dir, f"feature_drift_{timestamp}.png")
        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()

    def _plot_detailed_distributions(
        self,
        data_drift: Dict[str, Any],
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame,
        timestamp: str
    ):
        """
        Візуалізація детальних розподілів для ознак з високим дрейфом

        Args:
            data_drift: Результати виявлення дрейфу для ознак
            reference_data: Еталонний набір даних
            current_data: Поточний набір даних
            timestamp: Часова мітка для назви файлу
        """
        # Вибираємо ознаки з високим або середнім рівнем дрейфу
        drift_features = []
        for feature, info in data_drift.items():
            severity = info.get("severity", DriftSeverity.NO_DRIFT)
            if severity in [DriftSeverity.MEDIUM, DriftSeverity.HIGH]:
                drift_features.append(feature)

        if not drift_features:
            return

        # Обмежуємо кількість ознак для візуалізації
        max_features = 10
        if len(drift_features) > max_features:
            drift_features = drift_features[:max_features]

        # Для кожної ознаки створюємо окрему візуалізацію
        for feature in drift_features:
            if feature in reference_data.columns and feature in current_data.columns:
                # Перевіряємо тип даних
                is_numeric = np.issubdtype(reference_data[feature].dtype, np.number)

                fig, ax = plt.subplots(figsize=(12, 6))

                if is_numeric:
                    # Для числових ознак використовуємо KDE або гістограму
                    try:
                        sns.kdeplot(reference_data[feature], label="Еталонні дані", ax=ax)
                        sns.kdeplot(current_data[feature], label="Поточні дані", ax=ax)
                    except Exception:
                        # Якщо KDE не працює, використовуємо гістограму
                        sns.histplot(reference_data[feature], label="Еталонні дані", 
                                   alpha=0.5, ax=ax, stat="density", common_norm=True)
                        sns.histplot(current_data[feature], label="Поточні дані", 
                                   alpha=0.5, ax=ax, stat="density", common_norm=True)
                else:
                    # Для категоріальних ознак використовуємо гістограму
                    ref_counts = reference_data[feature].value_counts(normalize=True)
                    curr_counts = current_data[feature].value_counts(normalize=True)

                    # Об'єднуємо всі категорії
                    all_categories = list(set(ref_counts.index) | set(curr_counts.index))

                    # Обмежуємо кількість категорій, якщо їх забагато
                    max_categories = 15
                    if len(all_categories) > max_categories:
                        # Вибираємо найбільш часті категорії
                        top_ref = set(ref_counts.nlargest(max_categories // 2).index)
                        top_curr = set(curr_counts.nlargest(max_categories // 2).index)
                        all_categories = list(top_ref | top_curr)

                    # Створюємо DataFrame для візуалізації
                    plot_data = pd.DataFrame(index=all_categories)
                    plot_data["Еталонні дані"] = [ref_counts.get(cat, 0) for cat in all_categories]
                    plot_data["Поточні дані"] = [curr_counts.get(cat, 0) for cat in all_categories]

                    # Створюємо групований барплот
                    plot_data.plot(kind='bar', ax=ax)

                ax.set_title(f"Розподіл для ознаки '{feature}' (дрейф: {data_drift[feature]['score']:.4f})")
                ax.set_xlabel(feature)
                ax.set_ylabel("Щільність" if is_numeric else "Частота")
                ax.legend()

                # Зберігаємо діаграму
                output_file = os.path.join(self.output_dir, f"distribution_{feature}_{timestamp}.png")
                plt.tight_layout()
                plt.savefig(output_file)
                plt.close()

    def _plot_prediction_drift(self, prediction_drift: Dict[str, Any], timestamp: str):
        """
        Візуалізація дрейфу передбачень

        Args:
            prediction_drift: Результати виявлення дрейфу для передбачень
            timestamp: Часова мітка для назви файлу
        """
        if not prediction_drift:
            return

        score = prediction_drift.get("score", 0)
        severity = prediction_drift.get("severity", DriftSeverity.NO_DRIFT)
        details = prediction_drift.get("details", {})

        # Створюємо діаграму з рівнем дрейфу передбачень
        fig, ax = plt.subplots(figsize=(10, 6))

        # Визначаємо колір на основі рівня серйозності
        if severity == DriftSeverity.HIGH:
            color = "red"
        elif severity == DriftSeverity.MEDIUM:
            color = "orange"
        elif severity == DriftSeverity.LOW:
            color = "yellow"
        else:
            color = "green"

        # Горизонтальна шкала від 0 до 1
        ax.barh(["Дрейф передбачень"], [score], color=color)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Рівень дрейфу")
        ax.set_title(f"Дрейф передбачень: {score:.4f} ({severity})")

        # Додаємо вертикальні лінії для порогів серйозності
        ax.axvline(x=0.05, color='green', linestyle='--', alpha=0.7, label="Низький")
        ax.axvline(x=0.1, color='yellow', linestyle='--', alpha=0.7, label="Середній")
        ax.axvline(x=0.2, color='red', linestyle='--', alpha=0.7, label="Високий")
        ax.legend()

        # Зберігаємо діаграму
        output_file = os.path.join(self.output_dir, f"prediction_drift_{timestamp}.png")
        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()

        # Якщо є деталі зі статистичними змінами, створюємо додаткову візуалізацію
        stat_changes = details.get("statistic_changes", {})
        if stat_changes:
            fig, ax = plt.subplots(figsize=(10, 6))

            metrics = []
            changes = []
            colors = []

            for metric, change in stat_changes.items():
                metrics.append(metric)
                changes.append(change)

                # Визначаємо колір на основі величини зміни
                if change > 0.2:
                    colors.append("red")
                elif change > 0.1:
                    colors.append("orange")
                elif change > 0.05:
                    colors.append("yellow")
                else:
                    colors.append("green")

            # Сортуємо дані за величиною зміни
            sorted_indices = np.argsort(changes)
            metrics = [metrics[i] for i in sorted_indices]
            changes = [changes[i] for i in sorted_indices]
            colors = [colors[i] for i in sorted_indices]

            # Створюємо горизонтальну гістограму
            y_pos = np.arange(len(metrics))
            ax.barh(y_pos, changes, color=colors)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(metrics)
            ax.set_xlabel("Відносна зміна")
            ax.set_title("Зміни в статистиках передбачень")

            # Зберігаємо діаграму
            output_file = os.path.join(self.output_dir, f"prediction_stats_changes_{timestamp}.png")
            plt.tight_layout()
            plt.savefig(output_file)
            plt.close()
