# новлена версія для PR
# новлена версія для PR
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Р†РЅС‚РµРіСЂР°С†С–СЏ РґРµС‚РµРєС‚РѕСЂР° РґСЂРµР№С„Сѓ Р· Dagster pipeline
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

# РљРѕРЅС„С–РіСѓСЂР°С†С–СЏ РґР»СЏ РґРµС‚РµРєС‚РѕСЂР° РґСЂРµР№С„Сѓ
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
    Р—Р°РІР°РЅС‚Р°Р¶СѓС” РµС‚Р°Р»РѕРЅРЅС– РґР°РЅС– РґР»СЏ РІРёСЏРІР»РµРЅРЅСЏ РґСЂРµР№С„Сѓ
    """
    logger.info(f"Р—Р°РІР°РЅС‚Р°Р¶РµРЅРЅСЏ РµС‚Р°Р»РѕРЅРЅРёС… РґР°РЅРёС… Р· {config.reference_data_path}")

    if not os.path.exists(config.reference_data_path):
        raise FileNotFoundError(f"Р¤Р°Р№Р» РµС‚Р°Р»РѕРЅРЅРёС… РґР°РЅРёС… РЅРµ Р·РЅР°Р№РґРµРЅРѕ: {config.reference_data_path}")

    if config.reference_data_path.endswith('.csv'):
        reference_df = pd.read_csv(config.reference_data_path)
    elif config.reference_data_path.endswith('.parquet'):
        reference_df = pd.read_parquet(config.reference_data_path)
    else:
        raise ValueError(f"РќРµРїС–РґС‚СЂРёРјСѓРІР°РЅРёР№ С„РѕСЂРјР°С‚ С„Р°Р№Р»Сѓ: {config.reference_data_path}")

    return reference_df

@asset
def reference_predictions(config: DriftDetectorConfig) -> Optional[np.ndarray]:
    """
    Р—Р°РІР°РЅС‚Р°Р¶СѓС” РµС‚Р°Р»РѕРЅРЅС– РїРµСЂРµРґР±Р°С‡РµРЅРЅСЏ РґР»СЏ РІРёСЏРІР»РµРЅРЅСЏ РґСЂРµР№С„Сѓ
    """
    if not config.reference_predictions_path:
        logger.info("РЁР»СЏС… РґРѕ РµС‚Р°Р»РѕРЅРЅРёС… РїРµСЂРµРґР±Р°С‡РµРЅСЊ РЅРµ РІРєР°Р·Р°РЅРѕ, РїРѕРІРµСЂС‚Р°С”РјРѕ None")
        return None

    logger.info(f"Р—Р°РІР°РЅС‚Р°Р¶РµРЅРЅСЏ РµС‚Р°Р»РѕРЅРЅРёС… РїРµСЂРµРґР±Р°С‡РµРЅСЊ Р· {config.reference_predictions_path}")

    if not os.path.exists(config.reference_predictions_path):
        logger.warning(f"Р¤Р°Р№Р» РµС‚Р°Р»РѕРЅРЅРёС… РїРµСЂРµРґР±Р°С‡РµРЅСЊ РЅРµ Р·РЅР°Р№РґРµРЅРѕ: {config.reference_predictions_path}")
        return None

    if config.reference_predictions_path.endswith('.npy'):
        return np.load(config.reference_predictions_path)
    elif config.reference_predictions_path.endswith('.csv'):
        return pd.read_csv(config.reference_predictions_path).values
    elif config.reference_predictions_path.endswith('.pkl') or config.reference_predictions_path.endswith('.pickle'):
        with open(config.reference_predictions_path, 'rb') as f:
            return pickle.load(f)
    else:
        raise ValueError(f"РќРµРїС–РґС‚СЂРёРјСѓРІР°РЅРёР№ С„РѕСЂРјР°С‚ С„Р°Р№Р»Сѓ: {config.reference_predictions_path}")

@asset(deps=[reference_data, reference_predictions])
def drift_detector(config: DriftDetectorConfig, reference_data: pd.DataFrame, reference_predictions: Optional[np.ndarray]) -> DriftDetector:
    """
    РЎС‚РІРѕСЂСЋС” С‚Р° РЅР°Р»Р°С€С‚РѕРІСѓС” РґРµС‚РµРєС‚РѕСЂ РґСЂРµР№С„Сѓ
    """
    logger.info("РЎС‚РІРѕСЂРµРЅРЅСЏ РґРµС‚РµРєС‚РѕСЂР° РґСЂРµР№С„Сѓ")

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
    Р’РёСЏРІР»СЏС” РґСЂРµР№С„ РІ РїРѕС‚РѕС‡РЅРёС… РґР°РЅРёС… С‚Р° РїРµСЂРµРґР±Р°С‡РµРЅРЅСЏС…

    Args:
        detector: РќР°Р»Р°С€С‚РѕРІР°РЅРёР№ РґРµС‚РµРєС‚РѕСЂ РґСЂРµР№С„Сѓ
        current_data: РџРѕС‚РѕС‡РЅРёР№ РЅР°Р±С–СЂ РґР°РЅРёС…
        current_predictions: РџРѕС‚РѕС‡РЅС– РїРµСЂРµРґР±Р°С‡РµРЅРЅСЏ (РѕРїС†С–РѕРЅР°Р»СЊРЅРѕ)

    Returns:
        Р РµР·СѓР»СЊС‚Р°С‚Рё РІРёСЏРІР»РµРЅРЅСЏ РґСЂРµР№С„Сѓ
    """
    logger.info("Р’РёСЏРІР»РµРЅРЅСЏ РґСЂРµР№С„Сѓ РІ РїРѕС‚РѕС‡РЅРёС… РґР°РЅРёС…")

    # Р’РёРєРѕРЅСѓС”РјРѕ РІРёСЏРІР»РµРЅРЅСЏ РґСЂРµР№С„Сѓ
    drift_results = detector.detect_drift(current_data, current_predictions)

    # Р›РѕРіСѓРІР°РЅРЅСЏ СЂРµР·СѓР»СЊС‚Р°С‚С–РІ
    overall_score = drift_results["overall_drift"]["score"]
    overall_severity = drift_results["overall_drift"]["severity"]
    logger.info(f"Р’РёСЏРІР»РµРЅРѕ РґСЂРµР№С„ Р· Р·Р°РіР°Р»СЊРЅРѕСЋ РѕС†С–РЅРєРѕСЋ {overall_score:.4f} (СЂС–РІРµРЅСЊ: {overall_severity})")

    # Р›РѕРіСѓРІР°РЅРЅСЏ РґРµС‚Р°Р»РµР№ РґР»СЏ РѕР·РЅР°Рє Р· РІРёСЃРѕРєРёРј СЂС–РІРЅРµРј РґСЂРµР№С„Сѓ
    for feature, info in drift_results["data_drift"].items():
        if info["severity"] in [DriftSeverity.MEDIUM, DriftSeverity.HIGH]:
            logger.warning(f"Р’РёСЃРѕРєРёР№ СЂС–РІРµРЅСЊ РґСЂРµР№С„Сѓ РґР»СЏ РѕР·РЅР°РєРё '{feature}': {info['score']:.4f} (СЂС–РІРµРЅСЊ: {info['severity']})")

    # РЇРєС‰Рѕ С” РґСЂРµР№С„ Сѓ РїРµСЂРµРґР±Р°С‡РµРЅРЅСЏС…, С‚Р°РєРѕР¶ Р»РѕРіСѓС”РјРѕ
    if drift_results["prediction_drift"] is not None:
        pred_score = drift_results["prediction_drift"]["score"]
        pred_severity = drift_results["prediction_drift"]["severity"]
        if pred_severity in [DriftSeverity.MEDIUM, DriftSeverity.HIGH]:
            logger.warning(f"Р’РёСЃРѕРєРёР№ СЂС–РІРµРЅСЊ РґСЂРµР№С„Сѓ Сѓ РїРµСЂРµРґР±Р°С‡РµРЅРЅСЏС…: {pred_score:.4f} (СЂС–РІРµРЅСЊ: {pred_severity})")

    return drift_results

@op(out=Out(is_required=False))
def save_drift_results(config: DriftDetectorConfig, drift_results: Dict[str, Any]) -> Optional[str]:
    """
    Р—Р±РµСЂС–РіР°С” СЂРµР·СѓР»СЊС‚Р°С‚Рё РІРёСЏРІР»РµРЅРЅСЏ РґСЂРµР№С„Сѓ

    Args:
        config: РљРѕРЅС„С–РіСѓСЂР°С†С–СЏ РґРµС‚РµРєС‚РѕСЂР° РґСЂРµР№С„Сѓ
        drift_results: Р РµР·СѓР»СЊС‚Р°С‚Рё РІРёСЏРІР»РµРЅРЅСЏ РґСЂРµР№С„Сѓ

    Returns:
        РЁР»СЏС… РґРѕ Р·Р±РµСЂРµР¶РµРЅРѕРіРѕ С„Р°Р№Р»Сѓ СЂРµР·СѓР»СЊС‚Р°С‚С–РІ
    """
    # РЎС‚РІРѕСЂСЋС”РјРѕ РґРёСЂРµРєС‚РѕСЂС–СЋ РґР»СЏ СЂРµР·СѓР»СЊС‚Р°С‚С–РІ, СЏРєС‰Рѕ РІРѕРЅР° РЅРµ С–СЃРЅСѓС”
    os.makedirs(config.output_path, exist_ok=True)

    # Р¤РѕСЂРјСѓС”РјРѕ С–Рј'СЏ С„Р°Р№Р»Сѓ Р· РґР°С‚РѕСЋ С‚Р° С‡Р°СЃРѕРј
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(config.output_path, f"drift_results_{timestamp}.json")

    # Р—Р±РµСЂС–РіР°С”РјРѕ СЂРµР·СѓР»СЊС‚Р°С‚Рё РІ JSON
    with open(output_file, 'w') as f:
        json.dump(drift_results, f, indent=2, default=str)

    logger.info(f"Р РµР·СѓР»СЊС‚Р°С‚Рё РІРёСЏРІР»РµРЅРЅСЏ РґСЂРµР№С„Сѓ Р·Р±РµСЂРµР¶РµРЅРѕ РІ {output_file}")

    return output_file

@graph
def drift_detection_pipeline():
    """
    Р“СЂР°С„РѕРІРёР№ РєРѕРјРїРѕРЅРµРЅС‚ РґР»СЏ РІРёСЏРІР»РµРЅРЅСЏ РґСЂРµР№С„Сѓ РІ РґР°РЅРёС…
    """
    config = DriftDetectorConfig.from_resource()
    ref_data = reference_data(config)
    ref_predictions = reference_predictions(config)
    detector = drift_detector(config, ref_data, ref_predictions)

    # Р¦С– РѕРїРµСЂР°С†С–С— Р±СѓРґСѓС‚СЊ Р·'С”РґРЅР°РЅС– Р· РІС…С–РґРЅРёРјРё РґР°РЅРёРјРё Р· РѕСЃРЅРѕРІРЅРѕРіРѕ РїР°Р№РїР»Р°Р№РЅСѓ
    current_data = None  # Р‘СѓРґРµ РїРµСЂРµРґР°РЅРѕ Р· РїР°Р№РїР»Р°Р№РЅСѓ
    current_predictions = None  # Р‘СѓРґРµ РїРµСЂРµРґР°РЅРѕ Р· РїР°Р№РїР»Р°Р№РЅСѓ

    drift_results = detect_drift(detector, current_data, current_predictions)
    save_path = save_drift_results(config, drift_results)

    return drift_results, save_path

# РљРѕСЂРёСЃС‚СѓРІР°С‡С– РјРѕР¶СѓС‚СЊ РІРєР»СЋС‡РёС‚Рё С†РµР№ РіСЂР°С„РѕРІРёР№ РєРѕРјРїРѕРЅРµРЅС‚ Сѓ СЃРІС–Р№ РѕСЃРЅРѕРІРЅРёР№ РїР°Р№РїР»Р°Р№РЅ
# РЅР°РїСЂРёРєР»Р°Рґ:
# @job
# def main_pipeline():
#     data, predictions = process_and_predict()
#     drift_results, _ = drift_detection_pipeline(data, predictions)


