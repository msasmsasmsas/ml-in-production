# новлена версія для PR
# новлена версія для PR
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
РџСЂРёРєР»Р°Рґ РІРёРєРѕСЂРёСЃС‚Р°РЅРЅСЏ РґРµС‚РµРєС‚РѕСЂР° РґСЂРµР№С„Сѓ РґР°РЅРёС…
"""

import numpy as np
import pandas as pd
import logging
import json
import os
from datetime import datetime

from drift_detection import DriftDetector, DriftType, DriftSeverity
from visualization import DriftVisualizer

# РќР°Р»Р°С€С‚СѓРІР°РЅРЅСЏ Р»РѕРіСѓРІР°РЅРЅСЏ
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

def generate_synthetic_data(n_samples=1000, n_features=10, shift_factor=0.0, categorical_features=None):
    """
    Р“РµРЅРµСЂСѓС” СЃРёРЅС‚РµС‚РёС‡РЅС– РґР°РЅС– РґР»СЏ РґРµРјРѕРЅСЃС‚СЂР°С†С–С— РґРµС‚РµРєС‚РѕСЂР° РґСЂРµР№С„Сѓ

    Args:
        n_samples: РљС–Р»СЊРєС–СЃС‚СЊ Р·СЂР°Р·РєС–РІ
        n_features: РљС–Р»СЊРєС–СЃС‚СЊ РѕР·РЅР°Рє
        shift_factor: РљРѕРµС„С–С†С–С”РЅС‚ Р·СЃСѓРІСѓ РґР»СЏ СЃРёРјСѓР»СЏС†С–С— РґСЂРµР№С„Сѓ (0.0 = РЅРµРјР°С” РґСЂРµР№С„Сѓ, 1.0 = Р·РЅР°С‡РЅРёР№ РґСЂРµР№С„)
        categorical_features: РЎРїРёСЃРѕРє С–РЅРґРµРєСЃС–РІ РєР°С‚РµРіРѕСЂС–Р°Р»СЊРЅРёС… РѕР·РЅР°Рє

    Returns:
        DataFrame Р· СЃРёРЅС‚РµС‚РёС‡РЅРёРјРё РґР°РЅРёРјРё
    """
    if categorical_features is None:
        categorical_features = []

    # РЎС‚РІРѕСЂРµРЅРЅСЏ С‡РёСЃР»РѕРІРёС… РѕР·РЅР°Рє
    data = np.random.randn(n_samples, n_features) + shift_factor

    # Р”РѕРґР°РІР°РЅРЅСЏ РґСЂРµР№С„Сѓ РґРѕ РґРµСЏРєРёС… РѕР·РЅР°Рє
    if shift_factor > 0:
        # Р—СЃСѓРІ СЃРµСЂРµРґРЅСЊРѕРіРѕ Р·РЅР°С‡РµРЅРЅСЏ РґР»СЏ РїРµСЂС€РёС… РѕР·РЅР°Рє
        data[:, 0] += shift_factor * 2
        # Р—РјС–РЅР° РґРёСЃРїРµСЂСЃС–С— РґР»СЏ РґСЂСѓРіРёС… РѕР·РЅР°Рє
        data[:, 1] *= (1 + shift_factor)
        # Р—РјС–РЅР° СЂРѕР·РїРѕРґС–Р»Сѓ РґР»СЏ С‚СЂРµС‚С–С… РѕР·РЅР°Рє
        data[:, 2] = np.random.exponential(scale=1 + shift_factor, size=n_samples)

    # РЎС‚РІРѕСЂРµРЅРЅСЏ DataFrame
    df = pd.DataFrame(data, columns=[f"feature_{i}" for i in range(n_features)])

    # РџРµСЂРµС‚РІРѕСЂРµРЅРЅСЏ РґРµСЏРєРёС… РѕР·РЅР°Рє Сѓ РєР°С‚РµРіРѕСЂС–Р°Р»СЊРЅС–
    for idx in categorical_features:
        col_name = f"feature_{idx}"
        if col_name in df.columns:
            # РЎС‚РІРѕСЂРµРЅРЅСЏ РєР°С‚РµРіРѕСЂС–Р°Р»СЊРЅРѕС— РѕР·РЅР°РєРё Р· 5 РєР°С‚РµРіРѕСЂС–СЏРјРё
            categories = ["A", "B", "C", "D", "E"]

            # Р РѕР·РїРѕРґС–Р» РєР°С‚РµРіРѕСЂС–Р№
            if shift_factor == 0:
                # Р С–РІРЅРѕРјС–СЂРЅРёР№ СЂРѕР·РїРѕРґС–Р» РґР»СЏ РµС‚Р°Р»РѕРЅРЅРёС… РґР°РЅРёС…
                probs = np.ones(5) / 5
            else:
                # Р—РјС–С‰РµРЅРёР№ СЂРѕР·РїРѕРґС–Р» РґР»СЏ РґР°РЅРёС… Р· РґСЂРµР№С„РѕРј
                probs = np.array([0.1, 0.2, 0.4, 0.2, 0.1]) + shift_factor * np.array([0.2, 0.1, -0.3, 0.0, 0.0])
                probs = probs / probs.sum()  # РќРѕСЂРјР°Р»С–Р·Р°С†С–СЏ, С‰РѕР± СЃСѓРјР° Р±СѓР»Р° 1

            # Р“РµРЅРµСЂР°С†С–СЏ РєР°С‚РµРіРѕСЂС–Р№
            df[col_name] = np.random.choice(categories, size=n_samples, p=probs)

    return df

def main():
    # РџР°СЂР°РјРµС‚СЂРё РґР»СЏ СЃРёРЅС‚РµС‚РёС‡РЅРёС… РґР°РЅРёС…
    n_samples = 1000
    n_features = 10
    categorical_features = [3, 7, 9]  # Р†РЅРґРµРєСЃРё РєР°С‚РµРіРѕСЂС–Р°Р»СЊРЅРёС… РѕР·РЅР°Рє

    # РЎС‚РІРѕСЂРµРЅРЅСЏ РґРёСЂРµРєС‚РѕСЂС–С— РґР»СЏ СЂРµР·СѓР»СЊС‚Р°С‚С–РІ
    output_dir = "drift_example_results"
    os.makedirs(output_dir, exist_ok=True)

    # Р“РµРЅРµСЂР°С†С–СЏ РµС‚Р°Р»РѕРЅРЅРёС… РґР°РЅРёС… (Р±РµР· РґСЂРµР№С„Сѓ)
    logger.info("Р“РµРЅРµСЂР°С†С–СЏ РµС‚Р°Р»РѕРЅРЅРёС… РґР°РЅРёС…")
    reference_data = generate_synthetic_data(
        n_samples=n_samples,
        n_features=n_features,
        shift_factor=0.0,
        categorical_features=categorical_features
    )

    # РЎРёРјСѓР»СЏС†С–СЏ РїРµСЂРµРґР±Р°С‡РµРЅСЊ РґР»СЏ РµС‚Р°Р»РѕРЅРЅРёС… РґР°РЅРёС…
    logger.info("РЎРёРјСѓР»СЏС†С–СЏ РїРµСЂРµРґР±Р°С‡РµРЅСЊ РґР»СЏ РµС‚Р°Р»РѕРЅРЅРёС… РґР°РЅРёС…")
    reference_predictions = np.random.rand(n_samples)  # Р™РјРѕРІС–СЂРЅРѕСЃС‚С– РєР»Р°СЃСѓ 1
    reference_classes = (reference_predictions > 0.5).astype(int)  # Р‘С–РЅР°СЂРЅС– РєР»Р°СЃРё

    # Р С–РІРЅС– РґСЂРµР№С„Сѓ РґР»СЏ С‚РµСЃС‚СѓРІР°РЅРЅСЏ
    drift_levels = [0.0, 0.1, 0.3, 0.8]

    # РќР°Р»Р°С€С‚СѓРІР°РЅРЅСЏ РІС–Р·СѓР°Р»С–Р·Р°С‚РѕСЂР°
    visualizer = DriftVisualizer(output_dir=output_dir)

    # Р§РёСЃР»РѕРІС– С‚Р° РєР°С‚РµРіРѕСЂС–Р°Р»СЊРЅС– РєРѕР»РѕРЅРєРё
    numerical_columns = [f"feature_{i}" for i in range(n_features) if i not in categorical_features]
    categorical_columns = [f"feature_{i}" for i in categorical_features]

    # Р†РЅС–С†С–Р°Р»С–Р·Р°С†С–СЏ РґРµС‚РµРєС‚РѕСЂР° РґСЂРµР№С„Сѓ
    logger.info("Р†РЅС–С†С–Р°Р»С–Р·Р°С†С–СЏ РґРµС‚РµРєС‚РѕСЂР° РґСЂРµР№С„Сѓ")
    detector = DriftDetector(
        reference_data=reference_data,
        reference_predictions=reference_predictions,
        categorical_columns=categorical_columns,
        numerical_columns=numerical_columns,
        drift_threshold_low=0.05,
        drift_threshold_medium=0.1,
        drift_threshold_high=0.2
    )

    # РўРµСЃС‚СѓРІР°РЅРЅСЏ РґРµС‚РµРєС‚РѕСЂР° РЅР° СЂС–Р·РЅРёС… СЂС–РІРЅСЏС… РґСЂРµР№С„Сѓ
    for drift_level in drift_levels:
        logger.info(f"\nРўРµСЃС‚СѓРІР°РЅРЅСЏ РґСЂРµР№С„Сѓ Р· СЂС–РІРЅРµРј {drift_level}")

        # Р“РµРЅРµСЂР°С†С–СЏ РґР°РЅРёС… Р· РґСЂРµР№С„РѕРј
        current_data = generate_synthetic_data(
            n_samples=n_samples,
            n_features=n_features,
            shift_factor=drift_level,
            categorical_features=categorical_features
        )

        # РЎРёРјСѓР»СЏС†С–СЏ РїРµСЂРµРґР±Р°С‡РµРЅСЊ РґР»СЏ РїРѕС‚РѕС‡РЅРёС… РґР°РЅРёС…
        # Р”РѕРґР°С”РјРѕ РґСЂРµР№С„ Сѓ РїРµСЂРµРґР±Р°С‡РµРЅРЅСЏ Р·С– Р·Р±С–Р»СЊС€РµРЅРЅСЏРј Р№РјРѕРІС–СЂРЅРѕСЃС‚С– РєР»Р°СЃСѓ 1
        current_predictions = np.random.rand(n_samples) + drift_level * 0.2
        current_predictions = np.clip(current_predictions, 0, 1)  # РћР±РјРµР¶РµРЅРЅСЏ РІ РґС–Р°РїР°Р·РѕРЅС– [0, 1]
        current_classes = (current_predictions > 0.5).astype(int)

        # Р’РёСЏРІР»РµРЅРЅСЏ РґСЂРµР№С„Сѓ
        drift_results = detector.detect_drift(current_data, current_predictions)

        # Р’РёРІРµРґРµРЅРЅСЏ СЂРµР·СѓР»СЊС‚Р°С‚С–РІ
        overall_score = drift_results["overall_drift"]["score"]
        overall_severity = drift_results["overall_drift"]["severity"]
        logger.info(f"Р—Р°РіР°Р»СЊРЅРёР№ СЂС–РІРµРЅСЊ РґСЂРµР№С„Сѓ: {overall_score:.4f} ({overall_severity})")

        # Р—Р±РµСЂРµР¶РµРЅРЅСЏ СЂРµР·СѓР»СЊС‚Р°С‚С–РІ
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = os.path.join(output_dir, f"drift_results_level_{drift_level}_{timestamp}.json")
        with open(result_file, 'w') as f:
            json.dump(drift_results, f, indent=2, default=str)
        logger.info(f"Р РµР·СѓР»СЊС‚Р°С‚Рё Р·Р±РµСЂРµР¶РµРЅРѕ РІ {result_file}")

        # РЎС‚РІРѕСЂРµРЅРЅСЏ РІС–Р·СѓР°Р»С–Р·Р°С†С–Р№
        visualizer.visualize_drift_results(drift_results, reference_data, current_data)
        logger.info(f"Р’С–Р·СѓР°Р»С–Р·Р°С†С–С— Р·Р±РµСЂРµР¶РµРЅРѕ РІ {output_dir}")

if __name__ == "__main__":
    main()


