#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Інтеграція детектора дрейфу з Dagster pipeline
"""

import os
import json
import numpy as np
import pandas as pd
import pickle
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from dagster import (
    asset,
    AssetIn,
    AssetOut,
    Output,
    In,
    Out,
    op,
    job,
    IOManager,
    io_manager,
    MetadataValue,
    ResourceDefinition,
    graph,
    Config
)

from drift_detection import DriftDetector, DriftType, DriftSeverity

logger = logging.getLogger(__name__)

# Конфігурація для детектора дрейфу
class DriftDetectorConfig(Config):
    reference_data_path: str
    reference_predictions_path: Optional[str] = None
    categorical_columns: List[str] = []
    numerical_columns: List[str] = []
    drift_threshold_low: float = 0.05
    drift_threshold_medium: float = 0.1
    drift_threshold_high: float = 0.2
    output_path: str = "drift_results"

@asset
def reference_data(config: DriftDetectorConfig) -> pd.DataFrame:
    """
    Завантажує еталонні дані для виявлення дрейфу
    """
    logger.info(f"Завантаження еталонних даних з {config.reference_data_path}")

    if not os.path.exists(config.reference_data_path):
        raise FileNotFoundError(f"Файл еталонних даних не знайдено: {config.reference_data_path}")

    if config.reference_data_path.endswith('.csv'):
        reference_df = pd.read_csv(config.reference_data_path)
    elif config.reference_data_path.endswith('.parquet'):
        reference_df = pd.read_parquet(config.reference_data_path)
    else:
        raise ValueError(f"Непідтримуваний формат файлу: {config.reference_data_path}")

    return reference_df

@asset
def reference_predictions(config: DriftDetectorConfig) -> Optional[np.ndarray]:
    """
    Завантажує еталонні передбачення для виявлення дрейфу
    """
    if not config.reference_predictions_path:
        logger.info("Шлях до еталонних передбачень не вказано, повертаємо None")
        return None

    logger.info(f"Завантаження еталонних передбачень з {config.reference_predictions_path}")

    if not os.path.exists(config.reference_predictions_path):
        logger.warning(f"Файл еталонних передбачень не знайдено: {config.reference_predictions_path}")
        return None

    if config.reference_predictions_path.endswith('.npy'):
        return np.load(config.reference_predictions_path)
    elif config.reference_predictions_path.endswith('.csv'):
        return pd.read_csv(config.reference_predictions_path).values
    elif config.reference_predictions_path.endswith('.pkl') or config.reference_predictions_path.endswith('.pickle'):
        with open(config.reference_predictions_path, 'rb') as f:
            return pickle.load(f)
    else:
        raise ValueError(f"Непідтримуваний формат файлу: {config.reference_predictions_path}")

@asset(deps=[reference_data, reference_predictions])
def drift_detector(config: DriftDetectorConfig, reference_data: pd.DataFrame, reference_predictions: Optional[np.ndarray]) -> DriftDetector:
    """
    Створює та налаштовує детектор дрейфу
    """
    logger.info("Створення детектора дрейфу")

    detector = DriftDetector(
        reference_data=reference_data,
        reference_predictions=reference_predictions,
        categorical_columns=config.categorical_columns,
        numerical_columns=config.numerical_columns,
        drift_threshold_low=config.drift_threshold_low,
        drift_threshold_medium=config.drift_threshold_medium,
        drift_threshold_high=config.drift_threshold_high
    )

    return detector

@op(ins={"detector": In(), "current_data": In(), "current_predictions": In(is_required=False)})
def detect_drift(detector: DriftDetector, current_data: pd.DataFrame, current_predictions: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """
    Виявляє дрейф в поточних даних та передбаченнях

    Args:
        detector: Налаштований детектор дрейфу
        current_data: Поточний набір даних
        current_predictions: Поточні передбачення (опціонально)

    Returns:
        Результати виявлення дрейфу
    """
    logger.info("Виявлення дрейфу в поточних даних")

    # Виконуємо виявлення дрейфу
    drift_results = detector.detect_drift(current_data, current_predictions)

    # Логування результатів
    overall_score = drift_results["overall_drift"]["score"]
    overall_severity = drift_results["overall_drift"]["severity"]
    logger.info(f"Виявлено дрейф з загальною оцінкою {overall_score:.4f} (рівень: {overall_severity})")

    # Логування деталей для ознак з високим рівнем дрейфу
    for feature, info in drift_results["data_drift"].items():
        if info["severity"] in [DriftSeverity.MEDIUM, DriftSeverity.HIGH]:
            logger.warning(f"Високий рівень дрейфу для ознаки '{feature}': {info['score']:.4f} (рівень: {info['severity']})")

    # Якщо є дрейф у передбаченнях, також логуємо
    if drift_results["prediction_drift"] is not None:
        pred_score = drift_results["prediction_drift"]["score"]
        pred_severity = drift_results["prediction_drift"]["severity"]
        if pred_severity in [DriftSeverity.MEDIUM, DriftSeverity.HIGH]:
            logger.warning(f"Високий рівень дрейфу у передбаченнях: {pred_score:.4f} (рівень: {pred_severity})")

    return drift_results

@op(out=Out(is_required=False))
def save_drift_results(config: DriftDetectorConfig, drift_results: Dict[str, Any]) -> Optional[str]:
    """
    Зберігає результати виявлення дрейфу

    Args:
        config: Конфігурація детектора дрейфу
        drift_results: Результати виявлення дрейфу

    Returns:
        Шлях до збереженого файлу результатів
    """
    # Створюємо директорію для результатів, якщо вона не існує
    os.makedirs(config.output_path, exist_ok=True)

    # Формуємо ім'я файлу з датою та часом
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(config.output_path, f"drift_results_{timestamp}.json")

    # Зберігаємо результати в JSON
    with open(output_file, 'w') as f:
        json.dump(drift_results, f, indent=2, default=str)

    logger.info(f"Результати виявлення дрейфу збережено в {output_file}")

    return output_file

@graph
def drift_detection_pipeline():
    """
    Графовий компонент для виявлення дрейфу в даних
    """
    config = DriftDetectorConfig.from_resource()
    ref_data = reference_data(config)
    ref_predictions = reference_predictions(config)
    detector = drift_detector(config, ref_data, ref_predictions)

    # Ці операції будуть з'єднані з вхідними даними з основного пайплайну
    current_data = None  # Буде передано з пайплайну
    current_predictions = None  # Буде передано з пайплайну

    drift_results = detect_drift(detector, current_data, current_predictions)
    save_path = save_drift_results(config, drift_results)

    return drift_results, save_path

# Користувачі можуть включити цей графовий компонент у свій основний пайплайн
# наприклад:
# @job
# def main_pipeline():
#     data, predictions = process_and_predict()
#     drift_results, _ = drift_detection_pipeline(data, predictions)
