#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Dict, List, Optional, Union, Tuple, Any
import logging
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("distribution_shift")


class DistributionShiftDetector:
    """
    Класс для обнаружения изменений в распределении данных
    """
    def __init__(self, reference_data: pd.DataFrame):
        """
        Инициализация детектора с референсными данными
        
        Args:
            reference_data: Референсные данные (обучающая выборка)
        """
        self.reference_data = reference_data
        self.reference_stats = {}
        self.compute_reference_statistics()
    
    def compute_reference_statistics(self) -> None:
        """
        Вычисление статистик для референсных данных
        """
        for column in self.reference_data.columns:
            if pd.api.types.is_numeric_dtype(self.reference_data[column]):
                # Статистики для числовых признаков
                stats = {
                    "mean": self.reference_data[column].mean(),
                    "std": self.reference_data[column].std(),
                    "min": self.reference_data[column].min(),
                    "max": self.reference_data[column].max(),
                    "median": self.reference_data[column].median(),
                    "q1": self.reference_data[column].quantile(0.25),
                    "q3": self.reference_data[column].quantile(0.75),
                    "type": "numeric"
                }
            else:
                # Статистики для категориальных признаков
                value_counts = self.reference_data[column].value_counts(normalize=True)
                stats = {
                    "value_counts": value_counts.to_dict(),
                    "unique_count": len(value_counts),
                    "type": "categorical"
                }
            
            self.reference_stats[column] = stats
    
    def detect_numeric_shifts(self, new_data: pd.DataFrame, alpha: float = 0.05) -> Dict[str, Dict[str, Any]]:
        """
        Обнаружение изменений в распределении числовых признаков
        
        Args:
            new_data: Новые данные для проверки
            alpha: Уровень значимости для статистических тестов
            
        Returns:
            Словарь с результатами проверки для каждого числового признака
        """
        results = {}
        
        for column, stats in self.reference_stats.items():
            if stats["type"] != "numeric" or column not in new_data.columns:
                continue
            
            # Базовые статистики для новых данных
            new_mean = new_data[column].mean()
            new_std = new_data[column].std()
            
            # Относительное изменение среднего и стандартного отклонения
            mean_change = abs((new_mean - stats["mean"]) / stats["mean"]) if stats["mean"] != 0 else abs(new_mean)
            std_change = abs((new_std - stats["std"]) / stats["std"]) if stats["std"] != 0 else abs(new_std)
            
            # Статистический тест для проверки равенства распределений (Kolmogorov-Smirnov)
            ks_statistic, ks_pvalue = stats.ks_2samp(
                self.reference_data[column].dropna(), 
                new_data[column].dropna()
            )
            
            # Проверка на отклонение по t-тесту (сравнение средних)
            t_statistic, t_pvalue = stats.ttest_ind(
                self.reference_data[column].dropna(),
                new_data[column].dropna(),
                equal_var=False  # Не предполагаем равенство дисперсий (тест Уэлча)
            )
            
            # Определяем, есть ли значительное изменение в распределении
            has_shift = (ks_pvalue < alpha) or (mean_change > 0.1) or (std_change > 0.2)
            
            results[column] = {
                "reference_mean": stats["mean"],
                "reference_std": stats["std"],
                "new_mean": new_mean,
                "new_std": new_std,
                "mean_change": mean_change,
                "std_change": std_change,
                "ks_statistic": ks_statistic,
                "ks_pvalue": ks_pvalue,
                "t_statistic": t_statistic,
                "t_pvalue": t_pvalue,
                "has_shift": has_shift,
                "shift_severity": "high" if (mean_change > 0.3 or ks_pvalue < alpha/10) else 
                                 "medium" if (mean_change > 0.1 or ks_pvalue < alpha) else "low"
            }
        
        return results
    
    def detect_categorical_shifts(self, new_data: pd.DataFrame, alpha: float = 0.05) -> Dict[str, Dict[str, Any]]:
        """
        Обнаружение изменений в распределении категориальных признаков
        
        Args:
            new_data: Новые данные для проверки
            alpha: Уровень значимости для статистических тестов
            
        Returns:
            Словарь с результатами проверки для каждого категориального признака
        """
        results = {}
        
        for column, stats in self.reference_stats.items():
            if stats["type"] != "categorical" or column not in new_data.columns:
                continue
            
            # Распределение для новых данных
            new_value_counts = new_data[column].value_counts(normalize=True).to_dict()
            
            # Находим разницу в распределениях
            distribution_diff = {}
            all_values = set(stats["value_counts"].keys()) | set(new_value_counts.keys())
            
            for value in all_values:
                ref_prob = stats["value_counts"].get(value, 0)
                new_prob = new_value_counts.get(value, 0)
                distribution_diff[value] = new_prob - ref_prob
            
            # Статистический тест (хи-квадрат) для сравнения распределений
            # Для этого нам нужно привести данные к формату, подходящему для теста
            # Мы создаем таблицу сопряженности: строки - источники данных, столбцы - категории
            
            # Сначала собираем все уникальные значения
            all_values = sorted(list(all_values))
            
            # Создаем таблицу сопряженности
            contingency_table = []
            
            # Ряд для референсных данных
            ref_row = [self.reference_data[column].value_counts().get(val, 0) for val in all_values]
            # Ряд для новых данных
            new_row = [new_data[column].value_counts().get(val, 0) for val in all_values]
            
            contingency_table = np.array([ref_row, new_row])
            
            # Проверка на ненулевые суммы по строкам и столбцам
            # (для корректной работы хи-квадрат теста)
            valid_columns = (contingency_table.sum(axis=0) > 0)
            contingency_table = contingency_table[:, valid_columns]
            
            if contingency_table.shape[1] > 1 and np.all(contingency_table.sum(axis=1) > 0):
                chi2_statistic, chi2_pvalue, dof, expected = stats.chi2_contingency(contingency_table)
                
                # Оценка эффекта (V Крамера)
                n = contingency_table.sum()
                v_cramer = np.sqrt(chi2_statistic / (n * (min(contingency_table.shape) - 1)))
            else:
                chi2_statistic, chi2_pvalue, dof, expected = 0, 1, 0, np.array([])
                v_cramer = 0
            
            # Определяем, есть ли значительное изменение в распределении
            max_diff = max(abs(diff) for diff in distribution_diff.values())
            has_shift = (chi2_pvalue < alpha) or (max_diff > 0.1)
            
            results[column] = {
                "reference_distribution": stats["value_counts"],
                "new_distribution": new_value_counts,
                "distribution_diff": distribution_diff,
                "max_diff": max_diff,
                "chi2_statistic": chi2_statistic,
                "chi2_pvalue": chi2_pvalue,
                "v_cramer": v_cramer,
                "has_shift": has_shift,
                "shift_severity": "high" if (max_diff > 0.3 or chi2_pvalue < alpha/10) else
                                 "medium" if (max_diff > 0.1 or chi2_pvalue < alpha) else "low"
            }
        
        return results
    
    def detect_shifts(self, new_data: pd.DataFrame, alpha: float = 0.05) -> Dict[str, Dict[str, Any]]:
        """
        Обнаружение изменений в распределении всех признаков
        
        Args:
            new_data: Новые данные для проверки
            alpha: Уровень значимости для статистических тестов
            
        Returns:
            Словарь с результатами проверки для всех признаков
        """
        numeric_results = self.detect_numeric_shifts(new_data, alpha)
        categorical_results = self.detect_categorical_shifts(new_data, alpha)
        
        # Объединяем результаты
        results = {**numeric_results, **categorical_results}
        
        # Общая оценка изменения распределения
        shifts_detected = sum(1 for r in results.values() if r.get("has_shift", False))
        severity_counts = {
            "high": sum(1 for r in results.values() if r.get("shift_severity") == "high"),
            "medium": sum(1 for r in results.values() if r.get("shift_severity") == "medium"),
            "low": sum(1 for r in results.values() if r.get("shift_severity") == "low"),
        }
        
        results["_summary"] = {
            "total_features": len(results),
            "shifts_detected": shifts_detected,
            "shift_rate": shifts_detected / len(results) if results else 0,
            "severity_counts": severity_counts,
            "overall_severity": "high" if severity_counts["high"] > 0 else
                              "medium" if severity_counts["medium"] > 0 else
                              "low" if shifts_detected > 0 else "none"
        }
        
        return results
    
    def visualize_shifts(self, new_data: pd.DataFrame, results: Dict[str, Dict[str, Any]], 
                        output_dir: str = "shift_analysis") -> None:
        """
        Визуализация обнаруженных изменений в распределении
        
        Args:
            new_data: Новые данные
            results: Результаты анализа изменений в распределении
            output_dir: Директория для сохранения визуализаций
        """
        # Создаем директорию для выходных файлов, если ее нет
        os.makedirs(output_dir, exist_ok=True)
        
        # Создаем сводный отчет
        summary = results.get("_summary", {})
        shift_rate = summary.get("shift_rate", 0) * 100  # в процентах
        severity = summary.get("overall_severity", "none")
        
        plt.figure(figsize=(10, 6))
        
        # Основной график
        plt.subplot(1, 2, 1)
        labels = ['No Shift', 'Shift']
        sizes = [100 - shift_rate, shift_rate]
        colors = ['#66b3ff', '#ff9999']
        explode = (0, 0.1)
        
        plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
                shadow=True, startangle=90)
        plt.axis('equal')
        plt.title('Distribution Shift Analysis')
        
        # График по степени изменений
        plt.subplot(1, 2, 2)
        severity_counts = summary.get("severity_counts", {"high": 0, "medium": 0, "low": 0})
        labels = ['High', 'Medium', 'Low']
        sizes = [severity_counts.get("high", 0), severity_counts.get("medium", 0), severity_counts.get("low", 0)]
        colors = ['#ff9999', '#ffcc99', '#99ff99']
        
        if sum(sizes) > 0:
            plt.pie(sizes, labels=labels, colors=colors, autopct=lambda p: f'{int(p * sum(sizes) / 100)}',
                    shadow=True, startangle=90)
            plt.axis('equal')
            plt.title('Shift Severity')
        else:
            plt.text(0.5, 0.5, "No shifts detected", ha='center', va='center', fontsize=12)
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "distribution_shift_summary.png"))
        plt.close()
        
        # Визуализация изменений в отдельных признаках
        for column, column_results in results.items():
            if column == "_summary":
                continue
            
            if column in self.reference_stats and self.reference_stats[column]["type"] == "numeric":
                # Визуализация для числовых признаков
                plt.figure(figsize=(12, 6))
                
                # Гистограммы
                plt.subplot(1, 2, 1)
                sns.histplot(self.reference_data[column].dropna(), color='blue', label='Reference', alpha=0.5)
                sns.histplot(new_data[column].dropna(), color='red', label='New', alpha=0.5)
                plt.title(f'Distribution of {column}')
                plt.legend()
                
                # Ящики с усами
                plt.subplot(1, 2, 2)
                data_to_plot = [
                    self.reference_data[column].dropna(),
                    new_data[column].dropna()
                ]
                plt.boxplot(data_to_plot, labels=['Reference', 'New'])
                plt.title(f'Box Plot of {column}')
                
                # Добавляем информацию о результатах тестов
                plt.figtext(0.5, 0.01, 
                            f"KS Test p-value: {column_results.get('ks_pvalue', 'N/A'):.4f}, "
                            f"Mean Change: {column_results.get('mean_change', 'N/A'):.2f}, "
                            f"Shift: {column_results.get('shift_severity', 'N/A').upper()}",
                            ha="center", fontsize=12, bbox={"facecolor":"orange", "alpha":0.5, "pad":5})
                
                plt.tight_layout()
                plt.subplots_adjust(bottom=0.15)
                plt.savefig(os.path.join(output_dir, f"shift_{column}.png"))
                plt.close()
                
            elif column in self.reference_stats and self.reference_stats[column]["type"] == "categorical":
                # Визуализация для категориальных признаков
                plt.figure(figsize=(14, 7))
                
                # Столбчатая диаграмма распределений
                ref_dist = pd.Series(column_results.get("reference_distribution", {}))
                new_dist = pd.Series(column_results.get("new_distribution", {}))
                
                # Объединяем индексы для полного покрытия категорий
                all_categories = sorted(set(ref_dist.index) | set(new_dist.index))
                
                # Создаем DataFrame для визуализации
                plot_data = pd.DataFrame({
                    'Reference': [ref_dist.get(cat, 0) for cat in all_categories],
                    'New': [new_dist.get(cat, 0) for cat in all_categories]
                }, index=all_categories)
                
                # Строим столбчатую диаграмму
                ax = plot_data.plot(kind='bar', figsize=(14, 7))
                plt.title(f'Distribution Comparison for {column}')
                plt.xlabel('Categories')
                plt.ylabel('Frequency')
                plt.xticks(rotation=45)
                
                # Добавляем информацию о результатах тестов
                plt.figtext(0.5, 0.01, 
                            f"Chi2 Test p-value: {column_results.get('chi2_pvalue', 'N/A'):.4f}, "
                            f"V Cramer: {column_results.get('v_cramer', 'N/A'):.2f}, "
                            f"Max Diff: {column_results.get('max_diff', 'N/A'):.2f}, "
                            f"Shift: {column_results.get('shift_severity', 'N/A').upper()}",
                            ha="center", fontsize=12, bbox={"facecolor":"orange", "alpha":0.5, "pad":5})
                
                plt.tight_layout()
                plt.subplots_adjust(bottom=0.15)
                plt.savefig(os.path.join(output_dir, f"shift_{column}.png"))
                plt.close()


def main():
    """
    Пример использования детектора изменений в распределении
    """
    # Создание тестовых данных
    np.random.seed(42)
    
    # Референсные данные
    n_ref = 1000
    ref_data = {
        "feature1": np.random.normal(0, 1, n_ref),
        "feature2": np.random.uniform(-1, 1, n_ref),
        "feature3": np.random.choice(["A", "B", "C"], n_ref, p=[0.6, 0.3, 0.1]),
        "feature4": np.random.choice(["X", "Y"], n_ref, p=[0.5, 0.5])
    }
    reference_df = pd.DataFrame(ref_data)
    
    # Новые данные с измененным распределением
    n_new = 800
    new_data = {
        "feature1": np.random.normal(0.5, 1.2, n_new),  # Сдвиг среднего и увеличение дисперсии
        "feature2": np.random.uniform(-1, 1, n_new),    # Без изменений
        "feature3": np.random.choice(["A", "B", "C"], n_new, p=[0.4, 0.4, 0.2]),  # Изменение пропорций
        "feature4": np.random.choice(["X", "Y"], n_new, p=[0.5, 0.5])  # Без изменений
    }
    new_df = pd.DataFrame(new_data)
    
    # Создание и использование детектора
    detector = DistributionShiftDetector(reference_df)
    shift_results = detector.detect_shifts(new_df)
    
    # Вывод результатов
    summary = shift_results.get("_summary", {})
    logger.info(f"Shift analysis summary:")
    logger.info(f"- Total features analyzed: {summary.get('total_features', 0)}")
    logger.info(f"- Shifts detected: {summary.get('shifts_detected', 0)} ({summary.get('shift_rate', 0)*100:.1f}%)")
    logger.info(f"- Overall severity: {summary.get('overall_severity', 'N/A').upper()}")
    
    # Детальная информация по признакам с изменениями
    for column, results in shift_results.items():
        if column != "_summary" and results.get("has_shift", False):
            logger.info(f"\nShift detected in {column}:")
            logger.info(f"- Severity: {results.get('shift_severity', 'N/A').upper()}")
            
            if "ks_pvalue" in results:  # Числовой признак
                logger.info(f"- KS test p-value: {results.get('ks_pvalue', 'N/A'):.4f}")
                logger.info(f"- Mean change: {results.get('mean_change', 'N/A'):.2f}")
                logger.info(f"- Std change: {results.get('std_change', 'N/A'):.2f}")
            else:  # Категориальный признак
                logger.info(f"- Chi2 test p-value: {results.get('chi2_pvalue', 'N/A'):.4f}")
                logger.info(f"- V Cramer: {results.get('v_cramer', 'N/A'):.2f}")
                logger.info(f"- Max distribution difference: {results.get('max_diff', 'N/A'):.2f}")
    
    # Визуализация результатов
    detector.visualize_shifts(new_df, shift_results, "shift_analysis")
    logger.info(f"\nVisualizations saved to 'shift_analysis' directory")


if __name__ == "__main__":
    main()
