# новлена версія для PR
# новлена версія для PR
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
РњРѕРґСѓР»СЊ РґР»СЏ РІРёСЏРІР»РµРЅРЅСЏ РґСЂРµР№С„Сѓ РґР°РЅРёС… Сѓ РІС…С–РґРЅРёС… РѕР·РЅР°РєР°С… С‚Р° РІРёС…С–РґРЅРёС… РґР°РЅРёС… РјРѕРґРµР»С–
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
    РўРёРїРё РґСЂРµР№С„Сѓ РґР°РЅРёС…
    """
    DATA_DRIFT = "data_drift"  # Р”СЂРµР№С„ Сѓ РІС…С–РґРЅРёС… РґР°РЅРёС…
    PREDICTION_DRIFT = "prediction_drift"  # Р”СЂРµР№С„ Сѓ РїРµСЂРµРґР±Р°С‡РµРЅРЅСЏС…
    CONCEPT_DRIFT = "concept_drift"  # Р”СЂРµР№С„ Сѓ РІР·Р°С”РјРѕР·РІ'СЏР·РєСѓ РјС–Р¶ РІС…РѕРґРѕРј С– РІРёС…РѕРґРѕРј

class DriftSeverity(str, Enum):
    """
    Р С–РІРЅС– СЃРµСЂР№РѕР·РЅРѕСЃС‚С– РґСЂРµР№С„Сѓ
    """
    NO_DRIFT = "no_drift"  # Р”СЂРµР№С„Сѓ РЅРµРјР°С”
    LOW = "low"  # РќРёР·СЊРєРёР№ СЂС–РІРµРЅСЊ РґСЂРµР№С„Сѓ
    MEDIUM = "medium"  # РЎРµСЂРµРґРЅС–Р№ СЂС–РІРµРЅСЊ РґСЂРµР№С„Сѓ
    HIGH = "high"  # Р’РёСЃРѕРєРёР№ СЂС–РІРµРЅСЊ РґСЂРµР№С„Сѓ

class DriftDetector:
    """
    РљР»Р°СЃ РґР»СЏ РІРёСЏРІР»РµРЅРЅСЏ СЂС–Р·РЅРёС… С‚РёРїС–РІ РґСЂРµР№С„Сѓ РґР°РЅРёС…
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
        Р†РЅС–С†С–Р°Р»С–Р·Р°С†С–СЏ РґРµС‚РµРєС‚РѕСЂР° РґСЂРµР№С„Сѓ

        Args:
            reference_data: Р•С‚Р°Р»РѕРЅРЅРёР№ РЅР°Р±С–СЂ РґР°РЅРёС… РґР»СЏ РїРѕСЂС–РІРЅСЏРЅРЅСЏ
            reference_predictions: Р•С‚Р°Р»РѕРЅРЅС– РїРµСЂРµРґР±Р°С‡РµРЅРЅСЏ РјРѕРґРµР»С–
            categorical_columns: РЎРїРёСЃРѕРє РєР°С‚РµРіРѕСЂС–Р°Р»СЊРЅРёС… СЃС‚РѕРІРїС†С–РІ
            numerical_columns: РЎРїРёСЃРѕРє С‡РёСЃР»РѕРІРёС… СЃС‚РѕРІРїС†С–РІ
            drift_threshold_low: РџРѕСЂС–Рі РґР»СЏ РЅРёР·СЊРєРѕРіРѕ СЂС–РІРЅСЏ РґСЂРµР№С„Сѓ
            drift_threshold_medium: РџРѕСЂС–Рі РґР»СЏ СЃРµСЂРµРґРЅСЊРѕРіРѕ СЂС–РІРЅСЏ РґСЂРµР№С„Сѓ
            drift_threshold_high: РџРѕСЂС–Рі РґР»СЏ РІРёСЃРѕРєРѕРіРѕ СЂС–РІРЅСЏ РґСЂРµР№С„Сѓ
        """
        self.reference_data = reference_data
        self.reference_predictions = reference_predictions
        self.categorical_columns = categorical_columns or []
        self.numerical_columns = numerical_columns or []
        self.drift_threshold_low = drift_threshold_low
        self.drift_threshold_medium = drift_threshold_medium
        self.drift_threshold_high = drift_threshold_high

        # Р—Р±РµСЂС–РіР°С”РјРѕ СЃС‚Р°С‚РёСЃС‚РёРєРё РґР»СЏ РµС‚Р°Р»РѕРЅРЅРёС… РґР°РЅРёС…
        self.reference_stats = {}
        if reference_data is not None:
            self._calculate_reference_statistics()

        logger.info("Р†РЅС–С†С–Р°Р»С–Р·РѕРІР°РЅРѕ РґРµС‚РµРєС‚РѕСЂ РґСЂРµР№С„Сѓ РґР°РЅРёС…")

    def _calculate_reference_statistics(self):
        """
        РћР±С‡РёСЃР»РµРЅРЅСЏ СЃС‚Р°С‚РёСЃС‚РёРє РґР»СЏ РµС‚Р°Р»РѕРЅРЅРёС… РґР°РЅРёС…
        """
        logger.info("РћР±С‡РёСЃР»РµРЅРЅСЏ СЃС‚Р°С‚РёСЃС‚РёРє РґР»СЏ РµС‚Р°Р»РѕРЅРЅРёС… РґР°РЅРёС…")

        # Р”Р»СЏ С‡РёСЃР»РѕРІРёС… СЃС‚РѕРІРїС†С–РІ
        for col in self.numerical_columns:
            if col in self.reference_data.columns:
                self.reference_stats[col] = calculate_statistical_metrics(self.reference_data[col])

        # Р”Р»СЏ РєР°С‚РµРіРѕСЂС–Р°Р»СЊРЅРёС… СЃС‚РѕРІРїС†С–РІ
        for col in self.categorical_columns:
            if col in self.reference_data.columns:
                self.reference_stats[col] = calculate_distribution_metrics(self.reference_data[col])

        # РЇРєС‰Рѕ С” РµС‚Р°Р»РѕРЅРЅС– РїРµСЂРµРґР±Р°С‡РµРЅРЅСЏ
        if self.reference_predictions is not None:
            self.reference_stats["predictions"] = calculate_distribution_metrics(
                pd.Series(self.reference_predictions))

    def set_reference_data(self, reference_data: pd.DataFrame, reference_predictions: Optional[np.ndarray] = None):
        """
        Р’СЃС‚Р°РЅРѕРІР»РµРЅРЅСЏ Р°Р±Рѕ РѕРЅРѕРІР»РµРЅРЅСЏ РµС‚Р°Р»РѕРЅРЅРёС… РґР°РЅРёС…

        Args:
            reference_data: РќРѕРІРёР№ РµС‚Р°Р»РѕРЅРЅРёР№ РЅР°Р±С–СЂ РґР°РЅРёС…
            reference_predictions: РќРѕРІС– РµС‚Р°Р»РѕРЅРЅС– РїРµСЂРµРґР±Р°С‡РµРЅРЅСЏ
        """
        self.reference_data = reference_data
        if reference_predictions is not None:
            self.reference_predictions = reference_predictions

        # РџРµСЂРµСЂР°С…СѓРЅРѕРє СЃС‚Р°С‚РёСЃС‚РёРє
        self._calculate_reference_statistics()
        logger.info("РћРЅРѕРІР»РµРЅРѕ РµС‚Р°Р»РѕРЅРЅС– РґР°РЅС– С‚Р° СЃС‚Р°С‚РёСЃС‚РёРєРё")

    def detect_drift(
        self,
        current_data: pd.DataFrame,
        current_predictions: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Р’РёСЏРІР»РµРЅРЅСЏ РґСЂРµР№С„Сѓ РІ РїРѕС‚РѕС‡РЅРёС… РґР°РЅРёС… РїРѕСЂС–РІРЅСЏРЅРѕ Р· РµС‚Р°Р»РѕРЅРЅРёРјРё

        Args:
            current_data: РџРѕС‚РѕС‡РЅРёР№ РЅР°Р±С–СЂ РґР°РЅРёС… РґР»СЏ Р°РЅР°Р»С–Р·Сѓ
            current_predictions: РџРѕС‚РѕС‡РЅС– РїРµСЂРµРґР±Р°С‡РµРЅРЅСЏ РјРѕРґРµР»С– (РѕРїС†С–РѕРЅР°Р»СЊРЅРѕ)

        Returns:
            РЎР»РѕРІРЅРёРє Р· СЂРµР·СѓР»СЊС‚Р°С‚Р°РјРё РІРёСЏРІР»РµРЅРЅСЏ РґСЂРµР№С„Сѓ
        """
        if self.reference_data is None:
            raise ValueError("Р•С‚Р°Р»РѕРЅРЅС– РґР°РЅС– РЅРµ РІСЃС‚Р°РЅРѕРІР»РµРЅРѕ. Р’РёРєРѕСЂРёСЃС‚Р°Р№С‚Рµ set_reference_data()")

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

        # РџРµСЂРµРІС–СЂРєР° РґСЂРµР№С„Сѓ РґР»СЏ РєРѕР¶РЅРѕС— РѕР·РЅР°РєРё
        feature_drift_scores = []

        # Р”Р»СЏ С‡РёСЃР»РѕРІРёС… РѕР·РЅР°Рє
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

        # Р”Р»СЏ РєР°С‚РµРіРѕСЂС–Р°Р»СЊРЅРёС… РѕР·РЅР°Рє
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

        # РџРµСЂРµРІС–СЂРєР° РґСЂРµР№С„Сѓ РІ РїРµСЂРµРґР±Р°С‡РµРЅРЅСЏС…, СЏРєС‰Рѕ РІРѕРЅРё РЅР°РґР°РЅС–
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

        # РћР±С‡РёСЃР»РµРЅРЅСЏ Р·Р°РіР°Р»СЊРЅРѕРіРѕ СЂС–РІРЅСЏ РґСЂРµР№С„Сѓ
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
        Р’РёСЏРІР»РµРЅРЅСЏ РґСЂРµР№С„Сѓ РґР»СЏ С‡РёСЃР»РѕРІРёС… РѕР·РЅР°Рє

        Args:
            reference_data: Р•С‚Р°Р»РѕРЅРЅС– РґР°РЅС– РѕР·РЅР°РєРё
            current_data: РџРѕС‚РѕС‡РЅС– РґР°РЅС– РѕР·РЅР°РєРё

        Returns:
            Tuple Р· РѕС†С–РЅРєРѕСЋ РґСЂРµР№С„Сѓ С‚Р° РґРµС‚Р°Р»СЏРјРё
        """
        details = {}

        # РћР±С‡РёСЃР»РµРЅРЅСЏ СЃС‚Р°С‚РёСЃС‚РёС‡РЅРёС… РјРµС‚СЂРёРє РґР»СЏ РїРѕС‚РѕС‡РЅРёС… РґР°РЅРёС…
        current_stats = calculate_statistical_metrics(current_data)
        ref_stats = self.reference_stats.get(reference_data.name, {})

        # РџРѕСЂС–РІРЅСЏРЅРЅСЏ СЃС‚Р°С‚РёСЃС‚РёРє
        if ref_stats:
            for metric, value in current_stats.items():
                if metric in ref_stats:
                    # РћР±С‡РёСЃР»РµРЅРЅСЏ РІС–РґРЅРѕСЃРЅРѕС— Р·РјС–РЅРё
                    if ref_stats[metric] != 0:
                        rel_change = abs(value - ref_stats[metric]) / abs(ref_stats[metric])
                    else:
                        rel_change = 1.0 if value != 0 else 0.0

                    details[f"{metric}_change"] = rel_change

        # Р’РёРєРѕРЅР°РЅРЅСЏ СЃС‚Р°С‚РёСЃС‚РёС‡РЅРёС… С‚РµСЃС‚С–РІ
        # 1. РўРµСЃС‚ РљРѕР»РјРѕРіРѕСЂРѕРІР°-РЎРјРёСЂРЅРѕРІР°
        ks_stat, ks_pvalue = kolmogorov_smirnov_test(reference_data.values, current_data.values)
        details["ks_statistic"] = ks_stat
        details["ks_pvalue"] = ks_pvalue

        # 2. Р’С–РґСЃС‚Р°РЅСЊ Р’Р°СЃСЃРµСЂС€С‚РµР№РЅР°
        wd = wasserstein_distance(reference_data.values, current_data.values)
        details["wasserstein_distance"] = wd

        # 3. Р”РёРІРµСЂРіРµРЅС†С–СЏ Р”Р¶РµРЅСЃРµРЅР°-РЁРµРЅРЅРѕРЅР° (РґР»СЏ РґРёСЃРєСЂРµС‚РёР·РѕРІР°РЅРёС… РґР°РЅРёС…)
        js_div = jensen_shannon_divergence(reference_data.values, current_data.values)
        details["jensen_shannon_divergence"] = js_div

        # РћР±С‡РёСЃР»РµРЅРЅСЏ Р·Р°РіР°Р»СЊРЅРѕС— РѕС†С–РЅРєРё РґСЂРµР№С„Сѓ (РЅР° РѕСЃРЅРѕРІС– p-value С‚Р° РјРµС‚СЂРёРє РІС–РґСЃС‚Р°РЅС–)
        # РџРµСЂРµС‚РІРѕСЂРµРЅРЅСЏ p-value РІ РѕС†С–РЅРєСѓ РґСЂРµР№С„Сѓ (РјРµРЅС€Рµ p-value -> Р±С–Р»СЊС€РёР№ РґСЂРµР№С„)
        pvalue_score = 1.0 - min(ks_pvalue, 1.0)

        # РќРѕСЂРјР°Р»С–Р·Р°С†С–СЏ РІС–РґСЃС‚Р°РЅС– Р’Р°СЃСЃРµСЂС€С‚РµР№РЅР° (Р·Р°Р»РµР¶РёС‚СЊ РІС–Рґ РјР°СЃС€С‚Р°Р±Сѓ РґР°РЅРёС…)
        ref_range = ref_stats.get("max", 0) - ref_stats.get("min", 0)
        if ref_range > 0:
            wd_score = min(wd / ref_range, 1.0)
        else:
            wd_score = 0.0

        # РћР±'С”РґРЅР°РЅРЅСЏ РјРµС‚СЂРёРє РІ РѕРґРЅСѓ РѕС†С–РЅРєСѓ
        drift_score = 0.5 * pvalue_score + 0.3 * wd_score + 0.2 * js_div

        return drift_score, details

    def _detect_categorical_drift(
        self,
        reference_data: pd.Series,
        current_data: pd.Series
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Р’РёСЏРІР»РµРЅРЅСЏ РґСЂРµР№С„Сѓ РґР»СЏ РєР°С‚РµРіРѕСЂС–Р°Р»СЊРЅРёС… РѕР·РЅР°Рє

        Args:
            reference_data: Р•С‚Р°Р»РѕРЅРЅС– РґР°РЅС– РѕР·РЅР°РєРё
            current_data: РџРѕС‚РѕС‡РЅС– РґР°РЅС– РѕР·РЅР°РєРё

        Returns:
            Tuple Р· РѕС†С–РЅРєРѕСЋ РґСЂРµР№С„Сѓ С‚Р° РґРµС‚Р°Р»СЏРјРё
        """
        details = {}

        # РћР±С‡РёСЃР»РµРЅРЅСЏ СЂРѕР·РїРѕРґС–Р»С–РІ РєР°С‚РµРіРѕСЂС–Р№
        ref_dist = reference_data.value_counts(normalize=True).to_dict()
        current_dist = current_data.value_counts(normalize=True).to_dict()

        # РћР±'С”РґРЅР°РЅРЅСЏ РІСЃС–С… РєР°С‚РµРіРѕСЂС–Р№ Р· РѕР±РѕС… РЅР°Р±РѕСЂС–РІ РґР°РЅРёС…
        all_categories = set(ref_dist.keys()) | set(current_dist.keys())

        # Р—Р°РїРѕРІРЅРµРЅРЅСЏ РІС–РґСЃСѓС‚РЅС–С… РєР°С‚РµРіРѕСЂС–Р№ РЅСѓР»СЏРјРё
        ref_dist_complete = {cat: ref_dist.get(cat, 0) for cat in all_categories}
        current_dist_complete = {cat: current_dist.get(cat, 0) for cat in all_categories}

        # РџРµСЂРµС‚РІРѕСЂРµРЅРЅСЏ РІ РјР°СЃРёРІРё РґР»СЏ РѕР±С‡РёСЃР»РµРЅРЅСЏ РјРµС‚СЂРёРє
        ref_array = np.array(list(ref_dist_complete.values()))
        current_array = np.array(list(current_dist_complete.values()))

        # РћР±С‡РёСЃР»РµРЅРЅСЏ РґРёРІРµСЂРіРµРЅС†С–С— Р”Р¶РµРЅСЃРµРЅР°-РЁРµРЅРЅРѕРЅР°
        js_div = jensen_shannon_divergence(ref_array, current_array)
        details["jensen_shannon_divergence"] = js_div

        # РћР±С‡РёСЃР»РµРЅРЅСЏ С…С–-РєРІР°РґСЂР°С‚ С‚РµСЃС‚Сѓ РґР»СЏ РїРѕСЂС–РІРЅСЏРЅРЅСЏ СЂРѕР·РїРѕРґС–Р»С–РІ
        # РџРѕС‚СЂС–Р±РЅРѕ РїРµСЂРµС‚РІРѕСЂРёС‚Рё РІС–РґСЃРѕС‚РєРё Сѓ РєС–Р»СЊРєРѕСЃС‚С–
        ref_size = len(reference_data)
        current_size = len(current_data)

        # РџРµСЂРµРІС–СЂРєР°, С‡Рё РґРѕСЃС‚Р°С‚РЅСЊРѕ РґР°РЅРёС… РґР»СЏ С‚РµСЃС‚Сѓ С…С–-РєРІР°РґСЂР°С‚
        if ref_size >= 5 and current_size >= 5 and len(all_categories) > 1:
            try:
                ref_counts = np.array([ref_dist_complete[cat] * ref_size for cat in all_categories])
                current_counts = np.array([current_dist_complete[cat] * current_size for cat in all_categories])

                # Р—Р°РїРѕР±С–РіР°С”РјРѕ РїРѕРјРёР»РєР°Рј, СЏРєС‰Рѕ С” РЅСѓР»СЊРѕРІС– РѕС‡С–РєСѓРІР°РЅС– С‡Р°СЃС‚РѕС‚Рё
                mask = ref_counts > 0
                if sum(mask) > 1:  # РџРѕС‚СЂС–Р±РЅРѕ РїСЂРёРЅР°Р№РјРЅС– РґРІС– РЅРµРЅСѓР»СЊРѕРІС– РєР°С‚РµРіРѕСЂС–С—
                    chi2_stat, chi2_pvalue = stats.chisquare(current_counts[mask], ref_counts[mask])
                    details["chi2_statistic"] = chi2_stat
                    details["chi2_pvalue"] = chi2_pvalue

                    # РџРµСЂРµС‚РІРѕСЂРµРЅРЅСЏ p-value РІ РѕС†С–РЅРєСѓ РґСЂРµР№С„Сѓ
                    chi2_score = 1.0 - min(chi2_pvalue, 1.0)
                else:
                    chi2_score = 0.0
            except Exception as e:
                logger.warning(f"РџРѕРјРёР»РєР° РїСЂРё РѕР±С‡РёСЃР»РµРЅРЅС– С…С–-РєРІР°РґСЂР°С‚ С‚РµСЃС‚Сѓ: {str(e)}")
                chi2_score = 0.0
        else:
            chi2_score = 0.0

        # РћР±С‡РёСЃР»РµРЅРЅСЏ СЂС–Р·РЅРёС†С– РІ СЂРѕР·РїРѕРґС–Р»Р°С… РґР»СЏ РєРѕР¶РЅРѕС— РєР°С‚РµРіРѕСЂС–С—
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

        # РћР±С‡РёСЃР»РµРЅРЅСЏ РјР°РєСЃРёРјР°Р»СЊРЅРѕС— РІС–РґРЅРѕСЃРЅРѕС— Р·РјС–РЅРё РїРѕ РєР°С‚РµРіРѕСЂС–СЏС…
        max_rel_change = max([info["relative_change"] for info in category_changes.values()], default=0)
        details["max_relative_change"] = max_rel_change

        # РћР±С‡РёСЃР»РµРЅРЅСЏ Р·Р°РіР°Р»СЊРЅРѕС— РѕС†С–РЅРєРё РґСЂРµР№С„Сѓ РґР»СЏ РєР°С‚РµРіРѕСЂС–Р°Р»СЊРЅРёС… РґР°РЅРёС…
        # РљРѕРјР±С–РЅСѓС”РјРѕ JS РґРёРІРµСЂРіРµРЅС†С–СЋ, С…С–-РєРІР°РґСЂР°С‚ РѕС†С–РЅРєСѓ С‚Р° РјР°РєСЃРёРјР°Р»СЊРЅСѓ РІС–РґРЅРѕСЃРЅСѓ Р·РјС–РЅСѓ
        drift_score = 0.4 * js_div + 0.4 * chi2_score + 0.2 * min(max_rel_change, 1.0)

        return drift_score, details

    def _detect_prediction_drift(
        self,
        reference_predictions: np.ndarray,
        current_predictions: np.ndarray
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Р’РёСЏРІР»РµРЅРЅСЏ РґСЂРµР№С„Сѓ Сѓ РїРµСЂРµРґР±Р°С‡РµРЅРЅСЏС… РјРѕРґРµР»С–

        Args:
            reference_predictions: Р•С‚Р°Р»РѕРЅРЅС– РїРµСЂРµРґР±Р°С‡РµРЅРЅСЏ
            current_predictions: РџРѕС‚РѕС‡РЅС– РїРµСЂРµРґР±Р°С‡РµРЅРЅСЏ

        Returns:
            Tuple Р· РѕС†С–РЅРєРѕСЋ РґСЂРµР№С„Сѓ С‚Р° РґРµС‚Р°Р»СЏРјРё
        """
        details = {}

        # РџРµСЂРµС‚РІРѕСЂРµРЅРЅСЏ РІ РѕРґРЅРѕРІРёРјС–СЂРЅС– РјР°СЃРёРІРё, СЏРєС‰Рѕ РїРѕС‚СЂС–Р±РЅРѕ
        ref_preds = reference_predictions.flatten() if reference_predictions.ndim > 1 else reference_predictions
        current_preds = current_predictions.flatten() if current_predictions.ndim > 1 else current_predictions

        # РћР±С‡РёСЃР»РµРЅРЅСЏ СЃС‚Р°С‚РёСЃС‚РёС‡РЅРёС… РјРµС‚СЂРёРє
        ref_stats = calculate_statistical_metrics(pd.Series(ref_preds))
        current_stats = calculate_statistical_metrics(pd.Series(current_preds))

        # РџРѕСЂС–РІРЅСЏРЅРЅСЏ СЃС‚Р°С‚РёСЃС‚РёРє
        stat_changes = {}
        for metric, value in current_stats.items():
            if metric in ref_stats:
                # РћР±С‡РёСЃР»РµРЅРЅСЏ РІС–РґРЅРѕСЃРЅРѕС— Р·РјС–РЅРё
                if ref_stats[metric] != 0:
                    rel_change = abs(value - ref_stats[metric]) / abs(ref_stats[metric])
                else:
                    rel_change = 1.0 if value != 0 else 0.0

                stat_changes[f"{metric}_change"] = rel_change

        details["statistic_changes"] = stat_changes

        # РўРµСЃС‚ РљРѕР»РјРѕРіРѕСЂРѕРІР°-РЎРјРёСЂРЅРѕРІР°
        ks_stat, ks_pvalue = kolmogorov_smirnov_test(ref_preds, current_preds)
        details["ks_statistic"] = ks_stat
        details["ks_pvalue"] = ks_pvalue

        # Р’С–РґСЃС‚Р°РЅСЊ Р’Р°СЃСЃРµСЂС€С‚РµР№РЅР°
        wd = wasserstein_distance(ref_preds, current_preds)
        details["wasserstein_distance"] = wd

        # Р”РёРІРµСЂРіРµРЅС†С–СЏ Р”Р¶РµРЅСЃРµРЅР°-РЁРµРЅРЅРѕРЅР°
        js_div = jensen_shannon_divergence(ref_preds, current_preds)
        details["jensen_shannon_divergence"] = js_div

        # РћР±С‡РёСЃР»РµРЅРЅСЏ Р·Р°РіР°Р»СЊРЅРѕС— РѕС†С–РЅРєРё РґСЂРµР№С„Сѓ (РєРѕРјР±С–РЅР°С†С–СЏ СЂС–Р·РЅРёС… РјРµС‚СЂРёРє)
        # РџРµСЂРµС‚РІРѕСЂРµРЅРЅСЏ p-value РІ РѕС†С–РЅРєСѓ РґСЂРµР№С„Сѓ
        pvalue_score = 1.0 - min(ks_pvalue, 1.0)

        # РќРѕСЂРјР°Р»С–Р·Р°С†С–СЏ РІС–РґСЃС‚Р°РЅС– Р’Р°СЃСЃРµСЂС€С‚РµР№РЅР°
        ref_range = ref_stats.get("max", 0) - ref_stats.get("min", 0)
        if ref_range > 0:
            wd_score = min(wd / ref_range, 1.0)
        else:
            wd_score = 0.0

        # РЎРµСЂРµРґРЅСЏ РІС–РґРЅРѕСЃРЅР° Р·РјС–РЅР° РєР»СЋС‡РѕРІРёС… СЃС‚Р°С‚РёСЃС‚РёРє
        key_stats = ["mean_change", "std_change", "median_change"]
        mean_stat_change = np.mean([stat_changes.get(k, 0) for k in key_stats])

        # РћР±'С”РґРЅР°РЅРЅСЏ РјРµС‚СЂРёРє РІ РѕРґРЅСѓ РѕС†С–РЅРєСѓ
        drift_score = 0.3 * pvalue_score + 0.2 * wd_score + 0.3 * js_div + 0.2 * min(mean_stat_change, 1.0)

        return drift_score, details

    def _get_drift_severity(self, drift_score: float) -> DriftSeverity:
        """
        Р’РёР·РЅР°С‡РµРЅРЅСЏ СЂС–РІРЅСЏ СЃРµСЂР№РѕР·РЅРѕСЃС‚С– РґСЂРµР№С„Сѓ РЅР° РѕСЃРЅРѕРІС– РѕС†С–РЅРєРё

        Args:
            drift_score: РћС†С–РЅРєР° РґСЂРµР№С„Сѓ (0-1)

        Returns:
            Р С–РІРµРЅСЊ СЃРµСЂР№РѕР·РЅРѕСЃС‚С– РґСЂРµР№С„Сѓ
        """
        if drift_score < self.drift_threshold_low:
            return DriftSeverity.NO_DRIFT
        elif drift_score < self.drift_threshold_medium:
            return DriftSeverity.LOW
        elif drift_score < self.drift_threshold_high:
            return DriftSeverity.MEDIUM
        else:
            return DriftSeverity.HIGH


