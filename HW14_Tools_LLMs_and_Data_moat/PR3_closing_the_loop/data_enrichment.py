#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Модуль для створення датасету для розмітки з рішення моніторингу
"""

import os
import json
import time
import uuid
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dotenv import load_dotenv
from arize.pandas.logger import Client as ArizeClient
from labelbox import Client as LabelboxClient
from tqdm import tqdm
from evidently.model_monitoring import ModelMonitoring
from evidently.metrics import DataDriftTable, DataQualityPreset, ClassificationPerformancePreset
from evidently.report import Report
from evidently.dashboard import Dashboard
from evidently.dashboard.tabs import DataDriftTab, ClassificationPerformanceTab

# Завантаження змінних середовища
load_dotenv()

# Перевірка наявності ключів API
ARIZE_API_KEY = os.getenv("ARIZE_API_KEY")
ARIZE_SPACE_KEY = os.getenv("ARIZE_SPACE_KEY")
LABELBOX_API_KEY = os.getenv("LABELBOX_API_KEY")

# Перевірка наявності ключів
if not ARIZE_API_KEY or not ARIZE_SPACE_KEY:
    print("Попередження: ARIZE_API_KEY або ARIZE_SPACE_KEY не встановлені. Деякі функції можуть бути недоступні.")

if not LABELBOX_API_KEY:
    print("Попередження: LABELBOX_API_KEY не встановлений. Деякі функції можуть бути недоступні.")

# Шляхи до файлів
REFERENCE_DATA_PATH = "reference_data.csv"
CURRENT_DATA_PATH = "current_data.csv"
LABELING_TASKS_PATH = "labeling_tasks.json"
DASHBOARD_PATH = "drift_dashboard.html"

def generate_synthetic_monitoring_data(n_samples: int = 1000, drift_ratio: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Генерує синтетичні дані для імітації моніторингу моделі

    Args:
        n_samples: Кількість зразків
        drift_ratio: Частка зразків з дрейфом

    Returns:
        (reference_data, current_data) - еталонні та поточні дані
    """
    # Створюємо базові ознаки для еталонних даних
    reference_data = pd.DataFrame({
        "prediction_id": [f"ref_{i}" for i in range(n_samples)],
        "timestamp": [(datetime.now() - timedelta(days=30 + np.random.randint(0, 15))).isoformat() for _ in range(n_samples)],
        "green_level": np.random.normal(0.6, 0.1, n_samples),
        "leaf_area": np.random.normal(0.7, 0.15, n_samples),
        "moisture": np.random.normal(0.5, 0.1, n_samples),
        "temperature": np.random.normal(25, 3, n_samples),
        "humidity": np.random.normal(60, 10, n_samples),
        "leaf_spots": np.random.normal(0.2, 0.1, n_samples),
        "stem_discoloration": np.random.normal(0.15, 0.1, n_samples),
        "region_id": np.random.choice(["central", "north", "south", "east", "west"], n_samples),
        "crop_type": np.random.choice(["wheat", "corn", "potato", "sunflower", "barley"], n_samples),
        "image_url": [f"https://example.com/images/ref_{i}.jpg" for i in range(n_samples)]
    })

    # Додаємо передбачення для еталонних даних
    # 0=здорова, 1=хвороба, 2=шкідники, 3=бур'яни
    reference_data["prediction"] = np.random.choice([0, 1, 2, 3], n_samples, p=[0.4, 0.3, 0.2, 0.1])
    reference_data["actual"] = reference_data["prediction"]  # Для еталонних даних передбачення збігаються з фактичними значеннями

    # Створюємо копію для поточних даних
    current_data = reference_data.copy()
    current_data["prediction_id"] = [f"cur_{i}" for i in range(n_samples)]
    current_data["timestamp"] = [(datetime.now() - timedelta(days=np.random.randint(0, 15))).isoformat() for _ in range(n_samples)]
    current_data["image_url"] = [f"https://example.com/images/cur_{i}.jpg" for i in range(n_samples)]

    # Вносимо дрейф у поточні дані
    drift_samples = int(n_samples * drift_ratio)
    drift_indices = np.random.choice(n_samples, drift_samples, replace=False)

    # Дрейф у ознаках
    current_data.loc[drift_indices, "green_level"] = np.random.normal(0.45, 0.15, drift_samples)  # Зменшення зеленого кольору
    current_data.loc[drift_indices, "leaf_spots"] = np.random.normal(0.4, 0.2, drift_samples)  # Збільшення плям на листі
    current_data.loc[drift_indices, "stem_discoloration"] = np.random.normal(0.35, 0.2, drift_samples)  # Збільшення знебарвлення стебла

    # Дрейф у розподілі класів (більше випадків хвороб)
    current_data.loc[drift_indices, "prediction"] = np.random.choice([0, 1, 2, 3], drift_samples, p=[0.2, 0.5, 0.2, 0.1])

    # Імітуємо помилки моделі для зразків з дрейфом
    error_samples = int(drift_samples * 0.7)  # 70% зразків з дрейфом мають помилки
    error_indices = np.random.choice(drift_indices, error_samples, replace=False)

    # Генеруємо фактичні значення для зразків з помилками
    for idx in error_indices:
        pred = current_data.loc[idx, "prediction"]
        # Генеруємо інший клас, ніж передбачений
        actual_classes = [c for c in [0, 1, 2, 3] if c != pred]
        current_data.loc[idx, "actual"] = np.random.choice(actual_classes)

    return reference_data, current_data

def detect_drift_with_evidently(reference_data: pd.DataFrame, current_data: pd.DataFrame) -> Tuple[Report, List[Dict[str, Any]]]:
    """
    Виявляє дрейф даних за допомогою Evidently

    Args:
        reference_data: Еталонні дані
        current_data: Поточні дані

    Returns:
        (report, drift_samples) - звіт про дрейф та список зразків з дрейфом
    """
    # Вибираємо числові ознаки для аналізу
    numerical_features = [
        "green_level", "leaf_area", "moisture", "temperature", 
        "humidity", "leaf_spots", "stem_discoloration"
    ]

    # Вибираємо категоріальні ознаки
    categorical_features = ["region_id", "crop_type"]

    # Створюємо звіт про дрейф даних
    drift_report = Report(metrics=[
        DataDriftTable(column_names=numerical_features + categorical_features)
    ])

    # Створюємо звіт про продуктивність класифікації
    performance_report = Report(metrics=[
        ClassificationPerformancePreset()
    ])

    # Обчислюємо звіти
    drift_report.run(reference_data=reference_data, current_data=current_data)
    performance_report.run(
        reference_data=reference_data, 
        current_data=current_data,
        column_mapping={
            "target": "actual",
            "prediction": "prediction"
        }
    )

    # Створюємо інтерактивну інформаційну панель
    dashboard = Dashboard(tabs=[
        DataDriftTab(),
        ClassificationPerformanceTab()
    ])
    dashboard.calculate(reference_data, current_data, column_mapping={
        "target": "actual",
        "prediction": "prediction"
    })

    # Зберігаємо інформаційну панель як HTML
    dashboard.save(DASHBOARD_PATH)

    # Визначаємо зразки з дрейфом на основі метрик
    drift_metrics = drift_report.as_dict()

    # Виводимо основні метрики дрейфу
    drift_score = drift_metrics['metrics'][0]['result']['drift_score']
    print(f"Загальний показник дрейфу: {drift_score:.4f}")

    # Знаходимо ознаки з найбільшим дрейфом
    feature_drift = drift_metrics['metrics'][0]['result']['drift_by_columns']
    drifted_features = [f for f, info in feature_drift.items() if info['drift_detected']]
    print(f"Ознаки з виявленим дрейфом: {', '.join(drifted_features)}")

    # Визначаємо зразки з дрейфом
    drift_samples = []

    # Для кожного зразка в поточних даних перевіряємо, чи є дрейф в його ознаках
    for i, row in current_data.iterrows():
        sample_drift_score = 0

        # Підраховуємо показник дрейфу для зразка
        for feature in drifted_features:
            if feature in numerical_features:
                # Для числових ознак: порівнюємо з діапазоном еталонних даних
                ref_mean = reference_data[feature].mean()
                ref_std = reference_data[feature].std()
                z_score = abs((row[feature] - ref_mean) / ref_std)
                if z_score > 2.0:  # Значення відхиляється більше ніж на 2 стандартних відхилення
                    sample_drift_score += z_score / len(drifted_features)
            elif feature in categorical_features:
                # Для категоріальних ознак: перевіряємо, чи є значення рідкісним у еталонних даних
                ref_value_counts = reference_data[feature].value_counts(normalize=True)
                if row[feature] not in ref_value_counts or ref_value_counts[row[feature]] < 0.05:
                    sample_drift_score += 1.0 / len(drifted_features)

        # Перевіряємо, чи є помилка в передбаченні
        prediction_error = row["prediction"] != row["actual"]

        # Якщо зразок має значний дрейф або помилку передбачення, додаємо його до списку
        if sample_drift_score > 0.5 or prediction_error:
            drift_samples.append({
                "prediction_id": row["prediction_id"],
                "image_url": row["image_url"],
                "green_level": row["green_level"],
                "leaf_spots": row["leaf_spots"],
                "stem_discoloration": row["stem_discoloration"],
                "prediction": int(row["prediction"]),
                "actual": int(row["actual"]),
                "crop_type": row["crop_type"],
                "region_id": row["region_id"],
                "drift_score": sample_drift_score,
                "prediction_error": prediction_error
            })

    # Сортуємо зразки за показником дрейфу
    drift_samples.sort(key=lambda x: x["drift_score"], reverse=True)

    print(f"Виявлено {len(drift_samples)} зразків з дрейфом або помилками передбачення")
    print(f"Інформаційну панель збережено як {DASHBOARD_PATH}")

    return drift_report, drift_samples

def create_labeling_tasks(drift_samples: List[Dict[str, Any]], max_tasks: int = 100) -> List[Dict[str, Any]]:
    """
    Створює завдання для розмітки на основі зразків з дрейфом

    Args:
        drift_samples: Список зразків з дрейфом
        max_tasks: Максимальна кількість завдань

    Returns:
        Список завдань для розмітки
    """
    # Обмежуємо кількість завдань
    selected_samples = drift_samples[:min(len(drift_samples), max_tasks)]

    # Створюємо завдання для розмітки
    labeling_tasks = []

    for sample in selected_samples:
        task = {
            "external_id": sample["prediction_id"],
            "row_data": {
                "image_url": sample["image_url"],
                "crop_type": sample["crop_type"],
                "region_id": sample["region_id"],
                "green_level": float(sample["green_level"]),
                "leaf_spots": float(sample["leaf_spots"]),
                "stem_discoloration": float(sample["stem_discoloration"])
            },
            "model_prediction": int(sample["prediction"]),
            "annotation_options": [
                {"id": 0, "name": "Здорова"},
                {"id": 1, "name": "Хвороба"},
                {"id": 2, "name": "Шкідники"},
                {"id": 3, "name": "Бур'яни"}
            ],
            "priority": "high" if sample["prediction_error"] else "medium"
        }
        labeling_tasks.append(task)

    # Зберігаємо завдання для розмітки у файл
    with open(LABELING_TASKS_PATH, "w", encoding="utf-8") as f:
        json.dump(labeling_tasks, f, ensure_ascii=False, indent=2)

    print(f"Створено {len(labeling_tasks)} завдань для розмітки")
    print(f"Завдання збережено у файлі {LABELING_TASKS_PATH}")

    return labeling_tasks

def upload_to_labelbox(labeling_tasks: List[Dict[str, Any]], dataset_name: str = "crop_threats_drift") -> None:
    """
    Завантажує завдання для розмітки до Labelbox

    Args:
        labeling_tasks: Список завдань для розмітки
        dataset_name: Назва набору даних в Labelbox
    """
    if not LABELBOX_API_KEY:
        print("Неможливо завантажити завдання до Labelbox: відсутній API ключ")
        return

    try:
        # Ініціалізуємо клієнт Labelbox
        client = LabelboxClient(api_key=LABELBOX_API_KEY)

        # Створюємо набір даних
        dataset = client.create_dataset(name=f"{dataset_name}_{datetime.now().strftime('%Y%m%d')}")

        # Форматуємо дані для Labelbox
        uploads = []
        for task in labeling_tasks:
            uploads.append({
                "external_id": task["external_id"],
                "row_data": json.dumps(task["row_data"]),
                "attachments": [{
                    "type": "IMAGE",
                    "value": task["row_data"]["image_url"]
                }]
            })

        # Завантажуємо дані до Labelbox
        task = dataset.create_data_rows(uploads)
        task.wait_till_done()

        print(f"Успішно завантажено {len(labeling_tasks)} завдань до Labelbox")
        print(f"Набір даних: {dataset.name}")
        print(f"URL: https://app.labelbox.com/datasets/{dataset.uid}")

    except Exception as e:
        print(f"Помилка при завантаженні до Labelbox: {str(e)}")

def main():
    """
    Головна функція для демонстрації процесу створення датасету для розмітки
    """
    print("Генерація синтетичних даних для моніторингу...")
    reference_data, current_data = generate_synthetic_monitoring_data(n_samples=500, drift_ratio=0.3)

    # Зберігаємо дані у файли CSV для подальшого використання
    reference_data.to_csv(REFERENCE_DATA_PATH, index=False)
    current_data.to_csv(CURRENT_DATA_PATH, index=False)
    print(f"Еталонні дані збережено у файлі {REFERENCE_DATA_PATH}")
    print(f"Поточні дані збережено у файлі {CURRENT_DATA_PATH}")

    print("\nВиявлення дрейфу даних...")
    drift_report, drift_samples = detect_drift_with_evidently(reference_data, current_data)

    print("\nСтворення завдань для розмітки...")
    labeling_tasks = create_labeling_tasks(drift_samples, max_tasks=50)

    print("\nЗавантаження завдань до Labelbox...")
    upload_to_labelbox(labeling_tasks)

    print("\nГотово! Замкнутий цикл для покращення моделі створено.")
    print("Тепер ви можете:\n1. Використовувати дані з моніторингу для виявлення проблем")
    print("2. Створювати завдання для розмітки на основі проблемних зразків")
    print("3. Отримувати нові розмічені дані для покращення моделі")
    print("4. Продовжувати цикл для постійного покращення моделі")

if __name__ == "__main__":
    main()
