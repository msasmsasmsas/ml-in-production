# новлена версія для PR
# новлена версія для PR
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
РњРѕРґСѓР»СЊ Р· РјРµС‚СЂРёРєР°РјРё РґР»СЏ РІРёСЏРІР»РµРЅРЅСЏ РґСЂРµР№С„Сѓ РґР°РЅРёС…
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.spatial.distance import jensenshannon
from typing import Dict, Tuple, Any, List, Union

def calculate_statistical_metrics(data: pd.Series) -> Dict[str, float]:
    """
    РћР±С‡РёСЃР»РµРЅРЅСЏ СЃС‚Р°С‚РёСЃС‚РёС‡РЅРёС… РјРµС‚СЂРёРє РґР»СЏ С‡РёСЃР»РѕРІРѕС— СЃРµСЂС–С— РґР°РЅРёС…

    Args:
        data: РЎРµСЂС–СЏ РґР°РЅРёС… РґР»СЏ Р°РЅР°Р»С–Р·Сѓ

    Returns:
        РЎР»РѕРІРЅРёРє Р·С– СЃС‚Р°С‚РёСЃС‚РёС‡РЅРёРјРё РјРµС‚СЂРёРєР°РјРё
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
    РћР±С‡РёСЃР»РµРЅРЅСЏ РјРµС‚СЂРёРє СЂРѕР·РїРѕРґС–Р»Сѓ РґР»СЏ РєР°С‚РµРіРѕСЂС–Р°Р»СЊРЅРѕС— СЃРµСЂС–С— РґР°РЅРёС…

    Args:
        data: РЎРµСЂС–СЏ РґР°РЅРёС… РґР»СЏ Р°РЅР°Р»С–Р·Сѓ

    Returns:
        РЎР»РѕРІРЅРёРє Р· РјРµС‚СЂРёРєР°РјРё СЂРѕР·РїРѕРґС–Р»Сѓ
    """
    # РћР±С‡РёСЃР»РµРЅРЅСЏ СЂРѕР·РїРѕРґС–Р»Сѓ Р·РЅР°С‡РµРЅСЊ
    value_counts = data.value_counts(normalize=True)
    distribution = value_counts.to_dict()

    # РћР±С‡РёСЃР»РµРЅРЅСЏ РµРЅС‚СЂРѕРїС–С— СЂРѕР·РїРѕРґС–Р»Сѓ
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
    Р’РёРєРѕРЅР°РЅРЅСЏ С‚РµСЃС‚Сѓ РљРѕР»РјРѕРіРѕСЂРѕРІР°-РЎРјРёСЂРЅРѕРІР° РґР»СЏ РїРѕСЂС–РІРЅСЏРЅРЅСЏ РґРІРѕС… СЂРѕР·РїРѕРґС–Р»С–РІ

    Args:
        reference: Р•С‚Р°Р»РѕРЅРЅРёР№ РЅР°Р±С–СЂ РґР°РЅРёС…
        current: РџРѕС‚РѕС‡РЅРёР№ РЅР°Р±С–СЂ РґР°РЅРёС…

    Returns:
        Tuple Р· СЃС‚Р°С‚РёСЃС‚РёРєРѕСЋ С‚РµСЃС‚Сѓ С‚Р° p-value
    """
    try:
        # Р’РёРґР°Р»РµРЅРЅСЏ NaN Р·РЅР°С‡РµРЅСЊ
        reference_clean = reference[~np.isnan(reference)]
        current_clean = current[~np.isnan(current)]

        # Р’РёРєРѕРЅР°РЅРЅСЏ С‚РµСЃС‚Сѓ
        ks_stat, p_value = stats.ks_2samp(reference_clean, current_clean)
        return float(ks_stat), float(p_value)
    except Exception as e:
        # Р’ СЂР°Р·С– РїРѕРјРёР»РєРё РїРѕРІРµСЂС‚Р°С”РјРѕ РЅСѓР»СЊРѕРІС– Р·РЅР°С‡РµРЅРЅСЏ
        return 0.0, 1.0

def wasserstein_distance(reference: np.ndarray, current: np.ndarray) -> float:
    """
    РћР±С‡РёСЃР»РµРЅРЅСЏ РІС–РґСЃС‚Р°РЅС– Р’Р°СЃСЃРµСЂС€С‚РµР№РЅР° (Earth Mover's Distance) РјС–Р¶ РґРІРѕРјР° СЂРѕР·РїРѕРґС–Р»Р°РјРё

    Args:
        reference: Р•С‚Р°Р»РѕРЅРЅРёР№ РЅР°Р±С–СЂ РґР°РЅРёС…
        current: РџРѕС‚РѕС‡РЅРёР№ РЅР°Р±С–СЂ РґР°РЅРёС…

    Returns:
        Р’С–РґСЃС‚Р°РЅСЊ Р’Р°СЃСЃРµСЂС€С‚РµР№РЅР°
    """
    try:
        # Р’РёРґР°Р»РµРЅРЅСЏ NaN Р·РЅР°С‡РµРЅСЊ
        reference_clean = reference[~np.isnan(reference)]
        current_clean = current[~np.isnan(current)]

        # РћР±С‡РёСЃР»РµРЅРЅСЏ РІС–РґСЃС‚Р°РЅС–
        distance = stats.wasserstein_distance(reference_clean, current_clean)
        return float(distance)
    except Exception as e:
        # Р’ СЂР°Р·С– РїРѕРјРёР»РєРё РїРѕРІРµСЂС‚Р°С”РјРѕ РЅСѓР»СЊРѕРІРµ Р·РЅР°С‡РµРЅРЅСЏ
        return 0.0

def jensen_shannon_divergence(reference: np.ndarray, current: np.ndarray) -> float:
    """
    РћР±С‡РёСЃР»РµРЅРЅСЏ РґРёРІРµСЂРіРµРЅС†С–С— Р”Р¶РµРЅСЃРµРЅР°-РЁРµРЅРЅРѕРЅР° РјС–Р¶ РґРІРѕРјР° СЂРѕР·РїРѕРґС–Р»Р°РјРё

    Args:
        reference: Р•С‚Р°Р»РѕРЅРЅРёР№ РЅР°Р±С–СЂ РґР°РЅРёС…
        current: РџРѕС‚РѕС‡РЅРёР№ РЅР°Р±С–СЂ РґР°РЅРёС…

    Returns:
        Р”РёРІРµСЂРіРµРЅС†С–СЏ Р”Р¶РµРЅСЃРµРЅР°-РЁРµРЅРЅРѕРЅР°
    """
    try:
        # Р’РёРґР°Р»РµРЅРЅСЏ NaN Р·РЅР°С‡РµРЅСЊ
        reference_clean = reference[~np.isnan(reference)]
        current_clean = current[~np.isnan(current)]

        # Р”Р»СЏ РѕР±С‡РёСЃР»РµРЅРЅСЏ JS РґРёРІРµСЂРіРµРЅС†С–С— РїРѕС‚СЂС–Р±РЅС– РѕС†С–РЅРєРё РіСѓСЃС‚РёРЅРё СЂРѕР·РїРѕРґС–Р»Сѓ
        # Р’РёРєРѕСЂРёСЃС‚РѕРІСѓС”РјРѕ РіС–СЃС‚РѕРіСЂР°РјРё Р· РѕРґРЅР°РєРѕРІРёРјРё Р±С–РЅР°РјРё РґР»СЏ РѕР±РѕС… РЅР°Р±РѕСЂС–РІ РґР°РЅРёС…
        min_val = min(np.min(reference_clean), np.min(current_clean))
        max_val = max(np.max(reference_clean), np.max(current_clean))

        # РЇРєС‰Рѕ РґР°РЅС– РѕРґРЅР°РєРѕРІС–, РїРѕРІРµСЂС‚Р°С”РјРѕ 0
        if min_val == max_val:
            return 0.0

        # РљС–Р»СЊРєС–СЃС‚СЊ Р±С–РЅС–РІ РґР»СЏ РіС–СЃС‚РѕРіСЂР°РјРё (РєРѕСЂС–РЅСЊ Р· РєС–Р»СЊРєРѕСЃС‚С– РµР»РµРјРµРЅС‚С–РІ, Р°Р»Рµ РЅРµ РјРµРЅС€Рµ 5)
        n_bins = max(5, int(np.sqrt(min(len(reference_clean), len(current_clean)))))

        # РћР±С‡РёСЃР»РµРЅРЅСЏ РіС–СЃС‚РѕРіСЂР°Рј
        ref_hist, _ = np.histogram(reference_clean, bins=n_bins, range=(min_val, max_val), density=True)
        current_hist, _ = np.histogram(current_clean, bins=n_bins, range=(min_val, max_val), density=True)

        # РќРѕСЂРјР°Р»С–Р·Р°С†С–СЏ РіС–СЃС‚РѕРіСЂР°Рј, С‰РѕР± С—С… СЃСѓРјР° РґРѕСЂС–РІРЅСЋРІР°Р»Р° 1
        ref_hist = ref_hist / np.sum(ref_hist) if np.sum(ref_hist) > 0 else ref_hist
        current_hist = current_hist / np.sum(current_hist) if np.sum(current_hist) > 0 else current_hist

        # РћР±С‡РёСЃР»РµРЅРЅСЏ JS РґРёРІРµСЂРіРµРЅС†С–С—
        js_div = jensenshannon(ref_hist, current_hist)

        # JS РґРёРІРµСЂРіРµРЅС†С–СЏ РјРѕР¶Рµ Р±СѓС‚Рё NaN, СЏРєС‰Рѕ РѕРґРёРЅ Р· СЂРѕР·РїРѕРґС–Р»С–РІ РјР°С” РІСЃС– РЅСѓР»С–
        if np.isnan(js_div):
            return 0.0

        return float(js_div)
    except Exception as e:
        # Р’ СЂР°Р·С– РїРѕРјРёР»РєРё РїРѕРІРµСЂС‚Р°С”РјРѕ РЅСѓР»СЊРѕРІРµ Р·РЅР°С‡РµРЅРЅСЏ
        return 0.0

def chi_square_test(reference: np.ndarray, current: np.ndarray) -> Tuple[float, float]:
    """
    Р’РёРєРѕРЅР°РЅРЅСЏ С…С–-РєРІР°РґСЂР°С‚ С‚РµСЃС‚Сѓ РґР»СЏ РїРѕСЂС–РІРЅСЏРЅРЅСЏ РґРІРѕС… РєР°С‚РµРіРѕСЂС–Р°Р»СЊРЅРёС… СЂРѕР·РїРѕРґС–Р»С–РІ

    Args:
        reference: Р•С‚Р°Р»РѕРЅРЅРёР№ РЅР°Р±С–СЂ РґР°РЅРёС… (С‡Р°СЃС‚РѕС‚Рё)
        current: РџРѕС‚РѕС‡РЅРёР№ РЅР°Р±С–СЂ РґР°РЅРёС… (С‡Р°СЃС‚РѕС‚Рё)

    Returns:
        Tuple Р· СЃС‚Р°С‚РёСЃС‚РёРєРѕСЋ С‚РµСЃС‚Сѓ С‚Р° p-value
    """
    try:
        # РџРµСЂРµРІС–СЂРєР°, С‡Рё С” РЅРµРЅСѓР»СЊРѕРІС– С‡Р°СЃС‚РѕС‚Рё
        if np.sum(reference) == 0 or np.sum(current) == 0:
            return 0.0, 1.0

        # Р’РёРєРѕРЅР°РЅРЅСЏ С‚РµСЃС‚Сѓ
        chi2_stat, p_value = stats.chisquare(current, reference)
        return float(chi2_stat), float(p_value)
    except Exception as e:
        # Р’ СЂР°Р·С– РїРѕРјРёР»РєРё РїРѕРІРµСЂС‚Р°С”РјРѕ РЅСѓР»СЊРѕРІС– Р·РЅР°С‡РµРЅРЅСЏ
        return 0.0, 1.0


