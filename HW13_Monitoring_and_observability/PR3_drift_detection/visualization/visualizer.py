# новлена версія для PR
# новлена версія для PR
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
РњРѕРґСѓР»СЊ РґР»СЏ РІС–Р·СѓР°Р»С–Р·Р°С†С–С— СЂРµР·СѓР»СЊС‚Р°С‚С–РІ РІРёСЏРІР»РµРЅРЅСЏ РґСЂРµР№С„Сѓ РґР°РЅРёС…
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
    РљР»Р°СЃ РґР»СЏ РІС–Р·СѓР°Р»С–Р·Р°С†С–С— СЂРµР·СѓР»СЊС‚Р°С‚С–РІ РІРёСЏРІР»РµРЅРЅСЏ РґСЂРµР№С„Сѓ РґР°РЅРёС…
    """
    def __init__(self, output_dir: str = "drift_visualizations"):
        """
        Р†РЅС–С†С–Р°Р»С–Р·Р°С†С–СЏ РІС–Р·СѓР°Р»С–Р·Р°С‚РѕСЂР°

        Args:
            output_dir: Р”РёСЂРµРєС‚РѕСЂС–СЏ РґР»СЏ Р·Р±РµСЂРµР¶РµРЅРЅСЏ РІС–Р·СѓР°Р»С–Р·Р°С†С–Р№
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # РќР°Р»Р°С€С‚СѓРІР°РЅРЅСЏ СЃС‚РёР»СЋ РіСЂР°С„С–РєС–РІ
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_context("talk")

    def visualize_drift_results(self, drift_results: Dict[str, Any], reference_data: Optional[pd.DataFrame] = None, current_data: Optional[pd.DataFrame] = None):
        """
        РЎС‚РІРѕСЂРµРЅРЅСЏ РІС–Р·СѓР°Р»С–Р·Р°С†С–Р№ РЅР° РѕСЃРЅРѕРІС– СЂРµР·СѓР»СЊС‚Р°С‚С–РІ РІРёСЏРІР»РµРЅРЅСЏ РґСЂРµР№С„Сѓ

        Args:
            drift_results: Р РµР·СѓР»СЊС‚Р°С‚Рё РІРёСЏРІР»РµРЅРЅСЏ РґСЂРµР№С„Сѓ
            reference_data: Р•С‚Р°Р»РѕРЅРЅРёР№ РЅР°Р±С–СЂ РґР°РЅРёС… (РѕРїС†С–РѕРЅР°Р»СЊРЅРѕ)
            current_data: РџРѕС‚РѕС‡РЅРёР№ РЅР°Р±С–СЂ РґР°РЅРёС… (РѕРїС†С–РѕРЅР°Р»СЊРЅРѕ)
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Р’С–Р·СѓР°Р»С–Р·Р°С†С–СЏ Р·Р°РіР°Р»СЊРЅРѕРіРѕ СЂС–РІРЅСЏ РґСЂРµР№С„Сѓ
        self._plot_overall_drift(drift_results, timestamp)

        # Р’С–Р·СѓР°Р»С–Р·Р°С†С–СЏ РґСЂРµР№С„Сѓ РґР»СЏ РѕРєСЂРµРјРёС… РѕР·РЅР°Рє
        if "data_drift" in drift_results and drift_results["data_drift"]:
            self._plot_feature_drift(drift_results["data_drift"], timestamp)

            # РЇРєС‰Рѕ РґРѕСЃС‚СѓРїРЅС– РґР°РЅС–, СЃС‚РІРѕСЂСЋС”РјРѕ РґРµС‚Р°Р»СЊРЅС– РІС–Р·СѓР°Р»С–Р·Р°С†С–С— РґР»СЏ РѕР·РЅР°Рє
            if reference_data is not None and current_data is not None:
                self._plot_detailed_distributions(
                    drift_results["data_drift"],
                    reference_data,
                    current_data,
                    timestamp
                )

        # Р’С–Р·СѓР°Р»С–Р·Р°С†С–СЏ РґСЂРµР№С„Сѓ РїРµСЂРµРґР±Р°С‡РµРЅСЊ, СЏРєС‰Рѕ РІС–РЅ С”
        if "prediction_drift" in drift_results and drift_results["prediction_drift"] is not None:
            self._plot_prediction_drift(drift_results["prediction_drift"], timestamp)

    def _plot_overall_drift(self, drift_results: Dict[str, Any], timestamp: str):
        """
        Р’С–Р·СѓР°Р»С–Р·Р°С†С–СЏ Р·Р°РіР°Р»СЊРЅРѕРіРѕ СЂС–РІРЅСЏ РґСЂРµР№С„Сѓ

        Args:
            drift_results: Р РµР·СѓР»СЊС‚Р°С‚Рё РІРёСЏРІР»РµРЅРЅСЏ РґСЂРµР№С„Сѓ
            timestamp: Р§Р°СЃРѕРІР° РјС–С‚РєР° РґР»СЏ РЅР°Р·РІРё С„Р°Р№Р»Сѓ
        """
        overall_drift = drift_results.get("overall_drift", {})
        if not overall_drift:
            return

        score = overall_drift.get("score", 0)
        severity = overall_drift.get("severity", DriftSeverity.NO_DRIFT)

        # РЎС‚РІРѕСЂСЋС”РјРѕ РґС–Р°РіСЂР°РјСѓ Р· СЂС–РІРЅРµРј РґСЂРµР№С„Сѓ
        fig, ax = plt.subplots(figsize=(10, 6))

        # Р’РёР·РЅР°С‡Р°С”РјРѕ РєРѕР»С–СЂ РЅР° РѕСЃРЅРѕРІС– СЂС–РІРЅСЏ СЃРµСЂР№РѕР·РЅРѕСЃС‚С–
        if severity == DriftSeverity.HIGH:
            color = "red"
        elif severity == DriftSeverity.MEDIUM:
            color = "orange"
        elif severity == DriftSeverity.LOW:
            color = "yellow"
        else:
            color = "green"

        # Р“РѕСЂРёР·РѕРЅС‚Р°Р»СЊРЅР° С€РєР°Р»Р° РІС–Рґ 0 РґРѕ 1
        ax.barh(["Р—Р°РіР°Р»СЊРЅРёР№ РґСЂРµР№С„"], [score], color=color)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Р С–РІРµРЅСЊ РґСЂРµР№С„Сѓ")
        ax.set_title(f"Р—Р°РіР°Р»СЊРЅРёР№ СЂС–РІРµРЅСЊ РґСЂРµР№С„Сѓ: {score:.4f} ({severity})")

        # Р”РѕРґР°С”РјРѕ РІРµСЂС‚РёРєР°Р»СЊРЅС– Р»С–РЅС–С— РґР»СЏ РїРѕСЂРѕРіС–РІ СЃРµСЂР№РѕР·РЅРѕСЃС‚С–
        ax.axvline(x=0.05, color='green', linestyle='--', alpha=0.7, label="РќРёР·СЊРєРёР№")
        ax.axvline(x=0.1, color='yellow', linestyle='--', alpha=0.7, label="РЎРµСЂРµРґРЅС–Р№")
        ax.axvline(x=0.2, color='red', linestyle='--', alpha=0.7, label="Р’РёСЃРѕРєРёР№")
        ax.legend()

        # Р—Р±РµСЂС–РіР°С”РјРѕ РґС–Р°РіСЂР°РјСѓ
        output_file = os.path.join(self.output_dir, f"overall_drift_{timestamp}.png")
        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()

    def _plot_feature_drift(self, data_drift: Dict[str, Any], timestamp: str):
        """
        Р’С–Р·СѓР°Р»С–Р·Р°С†С–СЏ РґСЂРµР№С„Сѓ РґР»СЏ РѕРєСЂРµРјРёС… РѕР·РЅР°Рє

        Args:
            data_drift: Р РµР·СѓР»СЊС‚Р°С‚Рё РІРёСЏРІР»РµРЅРЅСЏ РґСЂРµР№С„Сѓ РґР»СЏ РѕР·РЅР°Рє
            timestamp: Р§Р°СЃРѕРІР° РјС–С‚РєР° РґР»СЏ РЅР°Р·РІРё С„Р°Р№Р»Сѓ
        """
        if not data_drift:
            return

        # РЎС‚РІРѕСЂСЋС”РјРѕ DataFrame Р· СЂРµР·СѓР»СЊС‚Р°С‚Р°РјРё РґР»СЏ РєРѕР¶РЅРѕС— РѕР·РЅР°РєРё
        features = []
        scores = []
        severities = []
        colors = []

        for feature, info in data_drift.items():
            features.append(feature)
            scores.append(info.get("score", 0))
            severity = info.get("severity", DriftSeverity.NO_DRIFT)
            severities.append(severity)

            # Р’РёР·РЅР°С‡Р°С”РјРѕ РєРѕР»С–СЂ РЅР° РѕСЃРЅРѕРІС– СЂС–РІРЅСЏ СЃРµСЂР№РѕР·РЅРѕСЃС‚С–
            if severity == DriftSeverity.HIGH:
                colors.append("red")
            elif severity == DriftSeverity.MEDIUM:
                colors.append("orange")
            elif severity == DriftSeverity.LOW:
                colors.append("yellow")
            else:
                colors.append("green")

        # РЎРѕСЂС‚СѓС”РјРѕ РґР°РЅС– Р·Р° СЂС–РІРЅРµРј РґСЂРµР№С„Сѓ
        sorted_indices = np.argsort(scores)
        features = [features[i] for i in sorted_indices]
        scores = [scores[i] for i in sorted_indices]
        severities = [severities[i] for i in sorted_indices]
        colors = [colors[i] for i in sorted_indices]

        # РЎС‚РІРѕСЂСЋС”РјРѕ РіРѕСЂРёР·РѕРЅС‚Р°Р»СЊРЅСѓ РіС–СЃС‚РѕРіСЂР°РјСѓ
        fig, ax = plt.subplots(figsize=(10, max(6, len(features) * 0.5)))

        y_pos = np.arange(len(features))
        ax.barh(y_pos, scores, color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Р С–РІРµРЅСЊ РґСЂРµР№С„Сѓ")
        ax.set_title("Р”СЂРµР№С„ РґР»СЏ РѕРєСЂРµРјРёС… РѕР·РЅР°Рє")

        # Р”РѕРґР°С”РјРѕ РІРµСЂС‚РёРєР°Р»СЊРЅС– Р»С–РЅС–С— РґР»СЏ РїРѕСЂРѕРіС–РІ СЃРµСЂР№РѕР·РЅРѕСЃС‚С–
        ax.axvline(x=0.05, color='green', linestyle='--', alpha=0.7, label="РќРёР·СЊРєРёР№")
        ax.axvline(x=0.1, color='yellow', linestyle='--', alpha=0.7, label="РЎРµСЂРµРґРЅС–Р№")
        ax.axvline(x=0.2, color='red', linestyle='--', alpha=0.7, label="Р’РёСЃРѕРєРёР№")
        ax.legend()

        # Р—Р±РµСЂС–РіР°С”РјРѕ РґС–Р°РіСЂР°РјСѓ
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
        Р’С–Р·СѓР°Р»С–Р·Р°С†С–СЏ РґРµС‚Р°Р»СЊРЅРёС… СЂРѕР·РїРѕРґС–Р»С–РІ РґР»СЏ РѕР·РЅР°Рє Р· РІРёСЃРѕРєРёРј РґСЂРµР№С„РѕРј

        Args:
            data_drift: Р РµР·СѓР»СЊС‚Р°С‚Рё РІРёСЏРІР»РµРЅРЅСЏ РґСЂРµР№С„Сѓ РґР»СЏ РѕР·РЅР°Рє
            reference_data: Р•С‚Р°Р»РѕРЅРЅРёР№ РЅР°Р±С–СЂ РґР°РЅРёС…
            current_data: РџРѕС‚РѕС‡РЅРёР№ РЅР°Р±С–СЂ РґР°РЅРёС…
            timestamp: Р§Р°СЃРѕРІР° РјС–С‚РєР° РґР»СЏ РЅР°Р·РІРё С„Р°Р№Р»Сѓ
        """
        # Р’РёР±РёСЂР°С”РјРѕ РѕР·РЅР°РєРё Р· РІРёСЃРѕРєРёРј Р°Р±Рѕ СЃРµСЂРµРґРЅС–Рј СЂС–РІРЅРµРј РґСЂРµР№С„Сѓ
        drift_features = []
        for feature, info in data_drift.items():
            severity = info.get("severity", DriftSeverity.NO_DRIFT)
            if severity in [DriftSeverity.MEDIUM, DriftSeverity.HIGH]:
                drift_features.append(feature)

        if not drift_features:
            return

        # РћР±РјРµР¶СѓС”РјРѕ РєС–Р»СЊРєС–СЃС‚СЊ РѕР·РЅР°Рє РґР»СЏ РІС–Р·СѓР°Р»С–Р·Р°С†С–С—
        max_features = 10
        if len(drift_features) > max_features:
            drift_features = drift_features[:max_features]

        # Р”Р»СЏ РєРѕР¶РЅРѕС— РѕР·РЅР°РєРё СЃС‚РІРѕСЂСЋС”РјРѕ РѕРєСЂРµРјСѓ РІС–Р·СѓР°Р»С–Р·Р°С†С–СЋ
        for feature in drift_features:
            if feature in reference_data.columns and feature in current_data.columns:
                # РџРµСЂРµРІС–СЂСЏС”РјРѕ С‚РёРї РґР°РЅРёС…
                is_numeric = np.issubdtype(reference_data[feature].dtype, np.number)

                fig, ax = plt.subplots(figsize=(12, 6))

                if is_numeric:
                    # Р”Р»СЏ С‡РёСЃР»РѕРІРёС… РѕР·РЅР°Рє РІРёРєРѕСЂРёСЃС‚РѕРІСѓС”РјРѕ KDE Р°Р±Рѕ РіС–СЃС‚РѕРіСЂР°РјСѓ
                    try:
                        sns.kdeplot(reference_data[feature], label="Р•С‚Р°Р»РѕРЅРЅС– РґР°РЅС–", ax=ax)
                        sns.kdeplot(current_data[feature], label="РџРѕС‚РѕС‡РЅС– РґР°РЅС–", ax=ax)
                    except Exception:
                        # РЇРєС‰Рѕ KDE РЅРµ РїСЂР°С†СЋС”, РІРёРєРѕСЂРёСЃС‚РѕРІСѓС”РјРѕ РіС–СЃС‚РѕРіСЂР°РјСѓ
                        sns.histplot(reference_data[feature], label="Р•С‚Р°Р»РѕРЅРЅС– РґР°РЅС–", 
                                   alpha=0.5, ax=ax, stat="density", common_norm=True)
                        sns.histplot(current_data[feature], label="РџРѕС‚РѕС‡РЅС– РґР°РЅС–", 
                                   alpha=0.5, ax=ax, stat="density", common_norm=True)
                else:
                    # Р”Р»СЏ РєР°С‚РµРіРѕСЂС–Р°Р»СЊРЅРёС… РѕР·РЅР°Рє РІРёРєРѕСЂРёСЃС‚РѕРІСѓС”РјРѕ РіС–СЃС‚РѕРіСЂР°РјСѓ
                    ref_counts = reference_data[feature].value_counts(normalize=True)
                    curr_counts = current_data[feature].value_counts(normalize=True)

                    # РћР±'С”РґРЅСѓС”РјРѕ РІСЃС– РєР°С‚РµРіРѕСЂС–С—
                    all_categories = list(set(ref_counts.index) | set(curr_counts.index))

                    # РћР±РјРµР¶СѓС”РјРѕ РєС–Р»СЊРєС–СЃС‚СЊ РєР°С‚РµРіРѕСЂС–Р№, СЏРєС‰Рѕ С—С… Р·Р°Р±Р°РіР°С‚Рѕ
                    max_categories = 15
                    if len(all_categories) > max_categories:
                        # Р’РёР±РёСЂР°С”РјРѕ РЅР°Р№Р±С–Р»СЊС€ С‡Р°СЃС‚С– РєР°С‚РµРіРѕСЂС–С—
                        top_ref = set(ref_counts.nlargest(max_categories // 2).index)
                        top_curr = set(curr_counts.nlargest(max_categories // 2).index)
                        all_categories = list(top_ref | top_curr)

                    # РЎС‚РІРѕСЂСЋС”РјРѕ DataFrame РґР»СЏ РІС–Р·СѓР°Р»С–Р·Р°С†С–С—
                    plot_data = pd.DataFrame(index=all_categories)
                    plot_data["Р•С‚Р°Р»РѕРЅРЅС– РґР°РЅС–"] = [ref_counts.get(cat, 0) for cat in all_categories]
                    plot_data["РџРѕС‚РѕС‡РЅС– РґР°РЅС–"] = [curr_counts.get(cat, 0) for cat in all_categories]

                    # РЎС‚РІРѕСЂСЋС”РјРѕ РіСЂСѓРїРѕРІР°РЅРёР№ Р±Р°СЂРїР»РѕС‚
                    plot_data.plot(kind='bar', ax=ax)

                ax.set_title(f"Р РѕР·РїРѕРґС–Р» РґР»СЏ РѕР·РЅР°РєРё '{feature}' (РґСЂРµР№С„: {data_drift[feature]['score']:.4f})")
                ax.set_xlabel(feature)
                ax.set_ylabel("Р©С–Р»СЊРЅС–СЃС‚СЊ" if is_numeric else "Р§Р°СЃС‚РѕС‚Р°")
                ax.legend()

                # Р—Р±РµСЂС–РіР°С”РјРѕ РґС–Р°РіСЂР°РјСѓ
                output_file = os.path.join(self.output_dir, f"distribution_{feature}_{timestamp}.png")
                plt.tight_layout()
                plt.savefig(output_file)
                plt.close()

    def _plot_prediction_drift(self, prediction_drift: Dict[str, Any], timestamp: str):
        """
        Р’С–Р·СѓР°Р»С–Р·Р°С†С–СЏ РґСЂРµР№С„Сѓ РїРµСЂРµРґР±Р°С‡РµРЅСЊ

        Args:
            prediction_drift: Р РµР·СѓР»СЊС‚Р°С‚Рё РІРёСЏРІР»РµРЅРЅСЏ РґСЂРµР№С„Сѓ РґР»СЏ РїРµСЂРµРґР±Р°С‡РµРЅСЊ
            timestamp: Р§Р°СЃРѕРІР° РјС–С‚РєР° РґР»СЏ РЅР°Р·РІРё С„Р°Р№Р»Сѓ
        """
        if not prediction_drift:
            return

        score = prediction_drift.get("score", 0)
        severity = prediction_drift.get("severity", DriftSeverity.NO_DRIFT)
        details = prediction_drift.get("details", {})

        # РЎС‚РІРѕСЂСЋС”РјРѕ РґС–Р°РіСЂР°РјСѓ Р· СЂС–РІРЅРµРј РґСЂРµР№С„Сѓ РїРµСЂРµРґР±Р°С‡РµРЅСЊ
        fig, ax = plt.subplots(figsize=(10, 6))

        # Р’РёР·РЅР°С‡Р°С”РјРѕ РєРѕР»С–СЂ РЅР° РѕСЃРЅРѕРІС– СЂС–РІРЅСЏ СЃРµСЂР№РѕР·РЅРѕСЃС‚С–
        if severity == DriftSeverity.HIGH:
            color = "red"
        elif severity == DriftSeverity.MEDIUM:
            color = "orange"
        elif severity == DriftSeverity.LOW:
            color = "yellow"
        else:
            color = "green"

        # Р“РѕСЂРёР·РѕРЅС‚Р°Р»СЊРЅР° С€РєР°Р»Р° РІС–Рґ 0 РґРѕ 1
        ax.barh(["Р”СЂРµР№С„ РїРµСЂРµРґР±Р°С‡РµРЅСЊ"], [score], color=color)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Р С–РІРµРЅСЊ РґСЂРµР№С„Сѓ")
        ax.set_title(f"Р”СЂРµР№С„ РїРµСЂРµРґР±Р°С‡РµРЅСЊ: {score:.4f} ({severity})")

        # Р”РѕРґР°С”РјРѕ РІРµСЂС‚РёРєР°Р»СЊРЅС– Р»С–РЅС–С— РґР»СЏ РїРѕСЂРѕРіС–РІ СЃРµСЂР№РѕР·РЅРѕСЃС‚С–
        ax.axvline(x=0.05, color='green', linestyle='--', alpha=0.7, label="РќРёР·СЊРєРёР№")
        ax.axvline(x=0.1, color='yellow', linestyle='--', alpha=0.7, label="РЎРµСЂРµРґРЅС–Р№")
        ax.axvline(x=0.2, color='red', linestyle='--', alpha=0.7, label="Р’РёСЃРѕРєРёР№")
        ax.legend()

        # Р—Р±РµСЂС–РіР°С”РјРѕ РґС–Р°РіСЂР°РјСѓ
        output_file = os.path.join(self.output_dir, f"prediction_drift_{timestamp}.png")
        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()

        # РЇРєС‰Рѕ С” РґРµС‚Р°Р»С– Р·С– СЃС‚Р°С‚РёСЃС‚РёС‡РЅРёРјРё Р·РјС–РЅР°РјРё, СЃС‚РІРѕСЂСЋС”РјРѕ РґРѕРґР°С‚РєРѕРІСѓ РІС–Р·СѓР°Р»С–Р·Р°С†С–СЋ
        stat_changes = details.get("statistic_changes", {})
        if stat_changes:
            fig, ax = plt.subplots(figsize=(10, 6))

            metrics = []
            changes = []
            colors = []

            for metric, change in stat_changes.items():
                metrics.append(metric)
                changes.append(change)

                # Р’РёР·РЅР°С‡Р°С”РјРѕ РєРѕР»С–СЂ РЅР° РѕСЃРЅРѕРІС– РІРµР»РёС‡РёРЅРё Р·РјС–РЅРё
                if change > 0.2:
                    colors.append("red")
                elif change > 0.1:
                    colors.append("orange")
                elif change > 0.05:
                    colors.append("yellow")
                else:
                    colors.append("green")

            # РЎРѕСЂС‚СѓС”РјРѕ РґР°РЅС– Р·Р° РІРµР»РёС‡РёРЅРѕСЋ Р·РјС–РЅРё
            sorted_indices = np.argsort(changes)
            metrics = [metrics[i] for i in sorted_indices]
            changes = [changes[i] for i in sorted_indices]
            colors = [colors[i] for i in sorted_indices]

            # РЎС‚РІРѕСЂСЋС”РјРѕ РіРѕСЂРёР·РѕРЅС‚Р°Р»СЊРЅСѓ РіС–СЃС‚РѕРіСЂР°РјСѓ
            y_pos = np.arange(len(metrics))
            ax.barh(y_pos, changes, color=colors)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(metrics)
            ax.set_xlabel("Р’С–РґРЅРѕСЃРЅР° Р·РјС–РЅР°")
            ax.set_title("Р—РјС–РЅРё РІ СЃС‚Р°С‚РёСЃС‚РёРєР°С… РїРµСЂРµРґР±Р°С‡РµРЅСЊ")

            # Р—Р±РµСЂС–РіР°С”РјРѕ РґС–Р°РіСЂР°РјСѓ
            output_file = os.path.join(self.output_dir, f"prediction_stats_changes_{timestamp}.png")
            plt.tight_layout()
            plt.savefig(output_file)
            plt.close()


