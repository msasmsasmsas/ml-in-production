# Updated version for PR
#!/usr/bin/env python
"""
Р†РЅСЃС‚СЂСѓРјРµРЅС‚ РґР»СЏ Р±РµРЅС‡РјР°СЂРєС–РЅРіСѓ РѕРєСЂРµРјРёС… РєРѕРјРїРѕРЅРµРЅС‚С–РІ СЃРµСЂРІРµСЂР° РјРѕРґРµР»РµР№ РјР°С€РёРЅРЅРѕРіРѕ РЅР°РІС‡Р°РЅРЅСЏ
"""

import os
import time
import json
import argparse
import statistics
import concurrent.futures
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision.models import resnet50, ResNet50_Weights

class ComponentBenchmarker:
    """
    РљР»Р°СЃ РґР»СЏ Р±РµРЅС‡РјР°СЂРєС–РЅРіСѓ РѕРєСЂРµРјРёС… РєРѕРјРїРѕРЅРµРЅС‚С–РІ С–РЅС„РµСЂРµРЅСЃСѓ РјРѕРґРµР»С–
    """
    def __init__(self, model_name='resnet50', device=None):
        """
        Р†РЅС–С†С–Р°Р»С–Р·Р°С†С–СЏ Р±РµРЅС‡РјР°СЂРєРµСЂР°

        РџР°СЂР°РјРµС‚СЂРё:
        -----------
        model_name: РЅР°Р·РІР° РјРѕРґРµР»С–
        device: РїСЂРёСЃС‚СЂС–Р№ РґР»СЏ РІРёРєРѕРЅР°РЅРЅСЏ (cuda Р°Р±Рѕ cpu)
        """
        self.model_name = model_name

        # Р’РёР·РЅР°С‡РµРЅРЅСЏ РїСЂРёСЃС‚СЂРѕСЋ
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        print(f"Р’РёРєРѕСЂРёСЃС‚Р°РЅРЅСЏ РїСЂРёСЃС‚СЂРѕСЋ: {self.device}")

        # Р—Р°РІР°РЅС‚Р°Р¶РµРЅРЅСЏ РјРѕРґРµР»С–
        self.model = self._load_model()

        # РЎС‚РІРѕСЂРµРЅРЅСЏ С‚СЂР°РЅСЃС„РѕСЂРјР°С†С–Р№ РґР»СЏ Р·РѕР±СЂР°Р¶РµРЅСЊ
        self.preprocessing = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def _load_model(self):
        """
        Р—Р°РІР°РЅС‚Р°Р¶РµРЅРЅСЏ РјРѕРґРµР»С–

        РџРѕРІРµСЂС‚Р°С”:
        -----------
        РјРѕРґРµР»СЊ PyTorch
        """
        if self.model_name == 'resnet50':
            model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        else:
            raise ValueError(f"РќРµРїС–РґС‚СЂРёРјСѓРІР°РЅР° РјРѕРґРµР»СЊ: {self.model_name}")

        model.to(self.device)
        model.eval()
        return model

    def benchmark_image_loading(self, image_path, iterations=100):
        """
        Р‘РµРЅС‡РјР°СЂРєС–РЅРі Р·Р°РІР°РЅС‚Р°Р¶РµРЅРЅСЏ Р·РѕР±СЂР°Р¶РµРЅСЊ

        РџР°СЂР°РјРµС‚СЂРё:
        -----------
        image_path: С€Р»СЏС… РґРѕ С„Р°Р№Р»Сѓ Р·РѕР±СЂР°Р¶РµРЅРЅСЏ
        iterations: РєС–Р»СЊРєС–СЃС‚СЊ С–С‚РµСЂР°С†С–Р№

        РџРѕРІРµСЂС‚Р°С”:
        -----------
        СЃР»РѕРІРЅРёРє Р· СЂРµР·СѓР»СЊС‚Р°С‚Р°РјРё
        """
        results = []

        for _ in range(iterations):
            start_time = time.time()
            with Image.open(image_path) as img:
                img_copy = img.copy()
            elapsed = time.time() - start_time
            results.append(elapsed)

        return self._calculate_stats(results, 'image_loading')

    def benchmark_preprocessing(self, image_path, iterations=100):
        """
        Р‘РµРЅС‡РјР°СЂРєС–РЅРі РїРѕРїРµСЂРµРґРЅСЊРѕС— РѕР±СЂРѕР±РєРё Р·РѕР±СЂР°Р¶РµРЅСЊ

        РџР°СЂР°РјРµС‚СЂРё:
        -----------
        image_path: С€Р»СЏС… РґРѕ С„Р°Р№Р»Сѓ Р·РѕР±СЂР°Р¶РµРЅРЅСЏ
        iterations: РєС–Р»СЊРєС–СЃС‚СЊ С–С‚РµСЂР°С†С–Р№

        РџРѕРІРµСЂС‚Р°С”:
        -----------
        СЃР»РѕРІРЅРёРє Р· СЂРµР·СѓР»СЊС‚Р°С‚Р°РјРё
        """
        with Image.open(image_path) as img:
            img_copy = img.copy()

        results = []

        for _ in range(iterations):
            start_time = time.time()
            tensor = self.preprocessing(img_copy)
            elapsed = time.time() - start_time
            results.append(elapsed)

        return self._calculate_stats(results, 'preprocessing')

    def benchmark_model_forward(self, image_path, batch_size=1, iterations=100):
        """
        Р‘РµРЅС‡РјР°СЂРєС–РЅРі РїСЂРѕС…РѕРґСѓ РІРїРµСЂРµРґ РјРѕРґРµР»С–

        РџР°СЂР°РјРµС‚СЂРё:
        -----------
        image_path: С€Р»СЏС… РґРѕ С„Р°Р№Р»Сѓ Р·РѕР±СЂР°Р¶РµРЅРЅСЏ
        batch_size: СЂРѕР·РјС–СЂ Р±Р°С‚С‡Сѓ
        iterations: РєС–Р»СЊРєС–СЃС‚СЊ С–С‚РµСЂР°С†С–Р№

        РџРѕРІРµСЂС‚Р°С”:
        -----------
        СЃР»РѕРІРЅРёРє Р· СЂРµР·СѓР»СЊС‚Р°С‚Р°РјРё
        """
        # РџС–РґРіРѕС‚РѕРІРєР° РІС…С–РґРЅРёС… РґР°РЅРёС…
        with Image.open(image_path) as img:
            tensor = self.preprocessing(img).unsqueeze(0).to(self.device)

        # РЎС‚РІРѕСЂРµРЅРЅСЏ Р±Р°С‚С‡Сѓ
        if batch_size > 1:
            tensor = tensor.repeat(batch_size, 1, 1, 1)

        results = []

        # РџСЂРѕРіСЂС–РІ GPU
        if self.device.type == 'cuda':
            print("РџСЂРѕРіСЂС–РІ GPU...")
            for _ in range(10):
                with torch.no_grad():
                    self.model(tensor)
            torch.cuda.synchronize()

        for _ in range(iterations):
            start_time = time.time()

            with torch.no_grad():
                outputs = self.model(tensor)

            if self.device.type == 'cuda':
                torch.cuda.synchronize()

            elapsed = time.time() - start_time
            results.append(elapsed)

        return self._calculate_stats(results, f'model_forward_batch{batch_size}')

    def benchmark_postprocessing(self, image_path, iterations=100):
        """
        Р‘РµРЅС‡РјР°СЂРєС–РЅРі РїС–СЃР»СЏРѕР±СЂРѕР±РєРё СЂРµР·СѓР»СЊС‚Р°С‚С–РІ

        РџР°СЂР°РјРµС‚СЂРё:
        -----------
        image_path: С€Р»СЏС… РґРѕ С„Р°Р№Р»Сѓ Р·РѕР±СЂР°Р¶РµРЅРЅСЏ
        iterations: РєС–Р»СЊРєС–СЃС‚СЊ С–С‚РµСЂР°С†С–Р№

        РџРѕРІРµСЂС‚Р°С”:
        -----------
        СЃР»РѕРІРЅРёРє Р· СЂРµР·СѓР»СЊС‚Р°С‚Р°РјРё
        """
        # РџС–РґРіРѕС‚РѕРІРєР° РІС…С–РґРЅРёС… РґР°РЅРёС…
        with Image.open(image_path) as img:
            tensor = self.preprocessing(img).unsqueeze(0).to(self.device)

        # РћС‚СЂРёРјР°РЅРЅСЏ РїСЂРѕРіРЅРѕР·Сѓ
        with torch.no_grad():
            outputs = self.model(tensor)

        results = []

        for _ in range(iterations):
            start_time = time.time()

            # РўРёРїРѕРІР° РїС–СЃР»СЏРѕР±СЂРѕР±РєР° РґР»СЏ РєР»Р°СЃРёС„С–РєР°С†С–С—
            probs = torch.nn.functional.softmax(outputs, dim=1)[0]
            top5_probs, top5_indices = torch.topk(probs, 5)

            # РљРѕРЅРІРµСЂС‚Р°С†С–СЏ Сѓ numpy С‚Р° С„РѕСЂРјР°С‚СѓРІР°РЅРЅСЏ СЂРµР·СѓР»СЊС‚Р°С‚С–РІ
            top5_probs = top5_probs.cpu().numpy()
            top5_indices = top5_indices.cpu().numpy()

            predictions = []
            for i, (idx, prob) in enumerate(zip(top5_indices, top5_probs)):
                predictions.append({'class_id': int(idx), 'score': float(prob)})

            elapsed = time.time() - start_time
            results.append(elapsed)

        return self._calculate_stats(results, 'postprocessing')

    def benchmark_end_to_end(self, image_path, iterations=100):
        """
        Р‘РµРЅС‡РјР°СЂРєС–РЅРі РїРѕРІРЅРѕРіРѕ РїСЂРѕС†РµСЃСѓ С–РЅС„РµСЂРµРЅСЃСѓ

        РџР°СЂР°РјРµС‚СЂРё:
        -----------
        image_path: С€Р»СЏС… РґРѕ С„Р°Р№Р»Сѓ Р·РѕР±СЂР°Р¶РµРЅРЅСЏ
        iterations: РєС–Р»СЊРєС–СЃС‚СЊ С–С‚РµСЂР°С†С–Р№

        РџРѕРІРµСЂС‚Р°С”:
        -----------
        СЃР»РѕРІРЅРёРє Р· СЂРµР·СѓР»СЊС‚Р°С‚Р°РјРё
        """
        results = []

        for _ in range(iterations):
            start_time = time.time()

            # Р—Р°РІР°РЅС‚Р°Р¶РµРЅРЅСЏ Р·РѕР±СЂР°Р¶РµРЅРЅСЏ
            with Image.open(image_path) as img:
                img_copy = img.copy()

            # РџРѕРїРµСЂРµРґРЅСЏ РѕР±СЂРѕР±РєР°
            tensor = self.preprocessing(img_copy).unsqueeze(0).to(self.device)

            # РџСЂРѕРіРЅРѕР·СѓРІР°РЅРЅСЏ
            with torch.no_grad():
                outputs = self.model(tensor)

            if self.device.type == 'cuda':
                torch.cuda.synchronize()

            # РџС–СЃР»СЏРѕР±СЂРѕР±РєР°
            probs = torch.nn.functional.softmax(outputs, dim=1)[0]
            top5_probs, top5_indices = torch.topk(probs, 5)

            top5_probs = top5_probs.cpu().numpy()
            top5_indices = top5_indices.cpu().numpy()

            predictions = []
            for i, (idx, prob) in enumerate(zip(top5_indices, top5_probs)):
                predictions.append({'class_id': int(idx), 'score': float(prob)})

            elapsed = time.time() - start_time
            results.append(elapsed)

        return self._calculate_stats(results, 'end_to_end')

    def benchmark_batch_sizes(self, image_path, batch_sizes=[1, 2, 4, 8, 16, 32], iterations=50):
        """
        Р‘РµРЅС‡РјР°СЂРєС–РЅРі СЂС–Р·РЅРёС… СЂРѕР·РјС–СЂС–РІ Р±Р°С‚С‡С–РІ

        РџР°СЂР°РјРµС‚СЂРё:
        -----------
        image_path: С€Р»СЏС… РґРѕ С„Р°Р№Р»Сѓ Р·РѕР±СЂР°Р¶РµРЅРЅСЏ
        batch_sizes: СЃРїРёСЃРѕРє СЂРѕР·РјС–СЂС–РІ Р±Р°С‚С‡С–РІ РґР»СЏ С‚РµСЃС‚СѓРІР°РЅРЅСЏ
        iterations: РєС–Р»СЊРєС–СЃС‚СЊ С–С‚РµСЂР°С†С–Р№ РґР»СЏ РєРѕР¶РЅРѕРіРѕ СЂРѕР·РјС–СЂСѓ Р±Р°С‚С‡Сѓ

        РџРѕРІРµСЂС‚Р°С”:
        -----------
        СЃРїРёСЃРѕРє СЃР»РѕРІРЅРёРєС–РІ Р· СЂРµР·СѓР»СЊС‚Р°С‚Р°РјРё
        """
        results = []

        for batch_size in batch_sizes:
            print(f"РўРµСЃС‚СѓРІР°РЅРЅСЏ СЂРѕР·РјС–СЂСѓ Р±Р°С‚С‡Сѓ {batch_size}...")
            result = self.benchmark_model_forward(image_path, batch_size=batch_size, iterations=iterations)
            results.append(result)

        return results

    def run_all_benchmarks(self, image_path, iterations=100):
        """
        Р—Р°РїСѓСЃРє РІСЃС–С… Р±РµРЅС‡РјР°СЂРєС–РІ

        РџР°СЂР°РјРµС‚СЂРё:
        -----------
        image_path: С€Р»СЏС… РґРѕ С„Р°Р№Р»Сѓ Р·РѕР±СЂР°Р¶РµРЅРЅСЏ
        iterations: РєС–Р»СЊРєС–СЃС‚СЊ С–С‚РµСЂР°С†С–Р№

        РџРѕРІРµСЂС‚Р°С”:
        -----------
        СЃР»РѕРІРЅРёРє Р· СЂРµР·СѓР»СЊС‚Р°С‚Р°РјРё РІСЃС–С… РєРѕРјРїРѕРЅРµРЅС‚С–РІ
        """
        results = {}

        print("Р‘РµРЅС‡РјР°СЂРєС–РЅРі Р·Р°РІР°РЅС‚Р°Р¶РµРЅРЅСЏ Р·РѕР±СЂР°Р¶РµРЅСЊ...")
        results['image_loading'] = self.benchmark_image_loading(image_path, iterations)

        print("Р‘РµРЅС‡РјР°СЂРєС–РЅРі РїРѕРїРµСЂРµРґРЅСЊРѕС— РѕР±СЂРѕР±РєРё...")
        results['preprocessing'] = self.benchmark_preprocessing(image_path, iterations)

        print("Р‘РµРЅС‡РјР°СЂРєС–РЅРі РїСЂРѕС…РѕРґСѓ РІРїРµСЂРµРґ РјРѕРґРµР»С–...")
        results['model_forward'] = self.benchmark_model_forward(image_path, iterations=iterations)

        print("Р‘РµРЅС‡РјР°СЂРєС–РЅРі РїС–СЃР»СЏРѕР±СЂРѕР±РєРё...")
        results['postprocessing'] = self.benchmark_postprocessing(image_path, iterations)

        print("Р‘РµРЅС‡РјР°СЂРєС–РЅРі РїРѕРІРЅРѕРіРѕ РїСЂРѕС†РµСЃСѓ С–РЅС„РµСЂРµРЅСЃСѓ...")
        results['end_to_end'] = self.benchmark_end_to_end(image_path, iterations)

        return results

    def _calculate_stats(self, times, name):
        """
        РћР±С‡РёСЃР»РµРЅРЅСЏ СЃС‚Р°С‚РёСЃС‚РёРєРё С‡Р°СЃСѓ РІРёРєРѕРЅР°РЅРЅСЏ

        РџР°СЂР°РјРµС‚СЂРё:
        -----------
        times: СЃРїРёСЃРѕРє С‡Р°СЃС–РІ РІРёРєРѕРЅР°РЅРЅСЏ
        name: РЅР°Р·РІР° РєРѕРјРїРѕРЅРµРЅС‚Р°

        РџРѕРІРµСЂС‚Р°С”:
        -----------
        СЃР»РѕРІРЅРёРє Р·С– СЃС‚Р°С‚РёСЃС‚РёРєРѕСЋ
        """
        # РљРѕРЅРІРµСЂС‚Р°С†С–СЏ Сѓ РјС–Р»С–СЃРµРєСѓРЅРґРё
        times_ms = [t * 1000 for t in times]

        return {
            'name': name,
            'iterations': len(times_ms),
            'min_ms': min(times_ms),
            'max_ms': max(times_ms),
            'mean_ms': statistics.mean(times_ms),
            'median_ms': statistics.median(times_ms),
            'p90_ms': np.percentile(times_ms, 90),
            'p95_ms': np.percentile(times_ms, 95),
            'p99_ms': np.percentile(times_ms, 99),
            'std_ms': statistics.stdev(times_ms) if len(times_ms) > 1 else 0,
            'raw_times_ms': times_ms
        }

def print_stats(stats):
    """
    Р’РёРІРµРґРµРЅРЅСЏ СЃС‚Р°С‚РёСЃС‚РёРєРё

    РџР°СЂР°РјРµС‚СЂРё:
    -----------
    stats: СЃР»РѕРІРЅРёРє Р·С– СЃС‚Р°С‚РёСЃС‚РёРєРѕСЋ
    """
    print(f"\nРЎС‚Р°С‚РёСЃС‚РёРєР° РґР»СЏ {stats['name']} ({stats['iterations']} С–С‚РµСЂР°С†С–Р№):")
    print(f"  РњС–РЅ: {stats['min_ms']:.3f} РјСЃ")
    print(f"  РњР°РєСЃ: {stats['max_ms']:.3f} РјСЃ")
    print(f"  РЎРµСЂРµРґРЅС”: {stats['mean_ms']:.3f} РјСЃ")
    print(f"  РњРµРґС–Р°РЅР°: {stats['median_ms']:.3f} РјСЃ")
    print(f"  P90: {stats['p90_ms']:.3f} РјСЃ")
    print(f"  P95: {stats['p95_ms']:.3f} РјСЃ")
    print(f"  P99: {stats['p99_ms']:.3f} РјСЃ")
    print(f"  РЎС‚Р°РЅРґР°СЂС‚РЅРµ РІС–РґС…РёР»РµРЅРЅСЏ: {stats['std_ms']:.3f} РјСЃ")

def save_results_json(results, output_file):
    """
    Р—Р±РµСЂС–РіР°С” СЂРµР·СѓР»СЊС‚Р°С‚Рё Сѓ JSON С„Р°Р№Р»

    РџР°СЂР°РјРµС‚СЂРё:
    -----------
    results: СЃР»РѕРІРЅРёРє Р· СЂРµР·СѓР»СЊС‚Р°С‚Р°РјРё
    output_file: С€Р»СЏС… РґРѕ РІРёС…С–РґРЅРѕРіРѕ С„Р°Р№Р»Сѓ
    """
    # РЎС‚РІРѕСЂРµРЅРЅСЏ РєРѕРїС–С— СЂРµР·СѓР»СЊС‚Р°С‚С–РІ Р±РµР· raw_times РґР»СЏ РєРѕРјРїР°РєС‚РЅРѕСЃС‚С–
    results_copy = {}

    for component, stats in results.items():
        if isinstance(stats, list):  # РґР»СЏ СЂРµР·СѓР»СЊС‚Р°С‚С–РІ batch_sizes
            results_copy[component] = []
            for stat in stats:
                stat_copy = stat.copy()
                if 'raw_times_ms' in stat_copy:
                    del stat_copy['raw_times_ms']
                results_copy[component].append(stat_copy)
        else:
            stats_copy = stats.copy()
            if 'raw_times_ms' in stats_copy:
                del stats_copy['raw_times_ms']
            results_copy[component] = stats_copy

    with open(output_file, 'w') as f:
        json.dump(results_copy, f, indent=2)

    print(f"Р РµР·СѓР»СЊС‚Р°С‚Рё Р·Р±РµСЂРµР¶РµРЅРѕ Сѓ {output_file}")

def plot_component_comparison(results, output_file=None):
    """
    РЎС‚РІРѕСЂСЋС” РіСЂР°С„С–РєРё РїРѕСЂС–РІРЅСЏРЅРЅСЏ РєРѕРјРїРѕРЅРµРЅС‚С–РІ

    РџР°СЂР°РјРµС‚СЂРё:
    -----------
    results: СЃР»РѕРІРЅРёРє Р· СЂРµР·СѓР»СЊС‚Р°С‚Р°РјРё
    output_file: С€Р»СЏС… РґРѕ РІРёС…С–РґРЅРѕРіРѕ С„Р°Р№Р»Сѓ (СЏРєС‰Рѕ None, РіСЂР°С„С–РєРё РІС–РґРѕР±СЂР°Р¶Р°СЋС‚СЊСЃСЏ)
    """
    components = ['image_loading', 'preprocessing', 'model_forward', 'postprocessing', 'end_to_end']
    available_components = [c for c in components if c in results]

    if not available_components:
        print("РќРµРґРѕСЃС‚Р°С‚РЅСЊРѕ РґР°РЅРёС… РґР»СЏ РїРѕР±СѓРґРѕРІРё РіСЂР°С„С–РєС–РІ")
        return

    # РЎС‚РІРѕСЂРµРЅРЅСЏ РіСЂР°С„С–РєС–РІ
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('РџРѕСЂС–РІРЅСЏРЅРЅСЏ РєРѕРјРїРѕРЅРµРЅС‚С–РІ С–РЅС„РµСЂРµРЅСЃСѓ', fontsize=16)

    # Р“СЂР°С„С–Рє 1: Р§Р°СЃ РІРёРєРѕРЅР°РЅРЅСЏ РєРѕРјРїРѕРЅРµРЅС‚С–РІ
    names = [results[c]['name'] for c in available_components]
    means = [results[c]['mean_ms'] for c in available_components]
    p95s = [results[c]['p95_ms'] for c in available_components]

    # РЎС‚РІРѕСЂРµРЅРЅСЏ РґРІРѕС… РіСЂР°С„С–РєС–РІ РЅР° РѕРґРЅРѕРјСѓ РїРѕР»РѕС‚РЅС–
    ax1.bar(names, means, label='РЎРµСЂРµРґРЅС–Р№ С‡Р°СЃ')
    ax1.bar(names, p95s, alpha=0.5, label='P95')
    ax1.set_title('Р§Р°СЃ РІРёРєРѕРЅР°РЅРЅСЏ РєРѕРјРїРѕРЅРµРЅС‚С–РІ')
    ax1.set_ylabel('Р§Р°СЃ (РјСЃ)')
    ax1.set_yscale('log')  # Р»РѕРіР°СЂРёС„РјС–С‡РЅР° С€РєР°Р»Р° РґР»СЏ РєСЂР°С‰РѕС— РІС–Р·СѓР°Р»С–Р·Р°С†С–С—
    ax1.legend()
    ax1.grid(axis='y', linestyle='--', alpha=0.7)

    # Р“СЂР°С„С–Рє 2: Р’С–РґРЅРѕСЃРЅРёР№ РІРєР»Р°Рґ РєРѕРјРїРѕРЅРµРЅС‚С–РІ
    # Р”Р»СЏ С†СЊРѕРіРѕ РіСЂР°С„С–РєР° РІРёРєРѕСЂРёСЃС‚РѕРІСѓС”РјРѕ С‚С–Р»СЊРєРё СЃРµСЂРµРґРЅС– Р·РЅР°С‡РµРЅРЅСЏ
    total_time = results['end_to_end']['mean_ms']
    other_components_time = sum([results[c]['mean_ms'] for c in available_components if c != 'end_to_end'])
    overhead_time = max(0, total_time - other_components_time)

    components_for_pie = available_components.copy()
    if 'end_to_end' in components_for_pie:
        components_for_pie.remove('end_to_end')

    labels = [results[c]['name'] for c in components_for_pie]
    sizes = [results[c]['mean_ms'] for c in components_for_pie]

    if overhead_time > 0:
        labels.append('overhead')
        sizes.append(overhead_time)

    ax2.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, shadow=True)
    ax2.axis('equal')  # СЂС–РІРЅС– РїСЂРѕРїРѕСЂС†С–С— РґР»СЏ РєСЂСѓРіРѕРІРѕС— РґС–Р°РіСЂР°РјРё
    ax2.set_title('Р’С–РґРЅРѕСЃРЅРёР№ РІРєР»Р°Рґ РєРѕРјРїРѕРЅРµРЅС‚С–РІ')

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if output_file:
        plt.savefig(output_file)
        print(f"Р“СЂР°С„С–Рє Р·Р±РµСЂРµР¶РµРЅРѕ Сѓ {output_file}")
    else:
        plt.show()

def plot_batch_size_comparison(batch_results, output_file=None):
    """
    РЎС‚РІРѕСЂСЋС” РіСЂР°С„С–РєРё РїРѕСЂС–РІРЅСЏРЅРЅСЏ СЂС–Р·РЅРёС… СЂРѕР·РјС–СЂС–РІ Р±Р°С‚С‡С–РІ

    РџР°СЂР°РјРµС‚СЂРё:
    -----------
    batch_results: СЃРїРёСЃРѕРє СЂРµР·СѓР»СЊС‚Р°С‚С–РІ РґР»СЏ СЂС–Р·РЅРёС… СЂРѕР·РјС–СЂС–РІ Р±Р°С‚С‡С–РІ
    output_file: С€Р»СЏС… РґРѕ РІРёС…С–РґРЅРѕРіРѕ С„Р°Р№Р»Сѓ (СЏРєС‰Рѕ None, РіСЂР°С„С–РєРё РІС–РґРѕР±СЂР°Р¶Р°СЋС‚СЊСЃСЏ)
    """
    if not batch_results:
        print("РќРµРґРѕСЃС‚Р°С‚РЅСЊРѕ РґР°РЅРёС… РґР»СЏ РїРѕР±СѓРґРѕРІРё РіСЂР°С„С–РєС–РІ")
        return

    # РћС‚СЂРёРјР°РЅРЅСЏ СЂРѕР·РјС–СЂС–РІ Р±Р°С‚С‡С–РІ Р· РЅР°Р·РІ
    batch_sizes = []
    mean_times = []
    throughputs = []  # Р·РѕР±СЂР°Р¶РµРЅСЊ РЅР° СЃРµРєСѓРЅРґСѓ

    for result in batch_results:
        # РћС‡С–РєСѓС”РјРѕ, С‰Рѕ РЅР°Р·РІР° РјР°С” С„РѕСЂРјР°С‚ 'model_forward_batchX'
        batch_size = int(result['name'].split('batch')[1])
        batch_sizes.append(batch_size)
        mean_times.append(result['mean_ms'])
        throughputs.append(batch_size * 1000 / result['mean_ms'])  # Р·РѕР±СЂР°Р¶РµРЅСЊ РЅР° СЃРµРєСѓРЅРґСѓ

    # РЎС‚РІРѕСЂРµРЅРЅСЏ РіСЂР°С„С–РєС–РІ
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('РџРѕСЂС–РІРЅСЏРЅРЅСЏ СЂРѕР·РјС–СЂС–РІ Р±Р°С‚С‡С–РІ', fontsize=16)

    # Р“СЂР°С„С–Рє 1: Р§Р°СЃ РІРёРєРѕРЅР°РЅРЅСЏ Р±Р°С‚С‡Сѓ
    ax1.plot(batch_sizes, mean_times, 'o-', label='РЎРµСЂРµРґРЅС–Р№ С‡Р°СЃ')
    ax1.set_title('Р§Р°СЃ РІРёРєРѕРЅР°РЅРЅСЏ Р±Р°С‚С‡Сѓ')
    ax1.set_xlabel('Р РѕР·РјС–СЂ Р±Р°С‚С‡Сѓ')
    ax1.set_ylabel('Р§Р°СЃ (РјСЃ)')
    ax1.grid(True)

    # Р“СЂР°С„С–Рє 2: РџСЂРѕРїСѓСЃРєРЅР° Р·РґР°С‚РЅС–СЃС‚СЊ
    ax2.plot(batch_sizes, throughputs, 'o-', label='РџСЂРѕРїСѓСЃРєРЅР° Р·РґР°С‚РЅС–СЃС‚СЊ')
    ax2.set_title('РџСЂРѕРїСѓСЃРєРЅР° Р·РґР°С‚РЅС–СЃС‚СЊ')
    ax2.set_xlabel('Р РѕР·РјС–СЂ Р±Р°С‚С‡Сѓ')
    ax2.set_ylabel('Р—РѕР±СЂР°Р¶РµРЅСЊ РЅР° СЃРµРєСѓРЅРґСѓ')
    ax2.grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if output_file:
        plt.savefig(output_file)
        print(f"Р“СЂР°С„С–Рє Р·Р±РµСЂРµР¶РµРЅРѕ Сѓ {output_file}")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='Р†РЅСЃС‚СЂСѓРјРµРЅС‚ РґР»СЏ Р±РµРЅС‡РјР°СЂРєС–РЅРіСѓ РєРѕРјРїРѕРЅРµРЅС‚С–РІ С–РЅС„РµСЂРµРЅСЃСѓ РјРѕРґРµР»С–')

    # РћСЃРЅРѕРІРЅС– РїР°СЂР°РјРµС‚СЂРё
    parser.add_argument('--image', type=str, required=True,
                        help='РЁР»СЏС… РґРѕ С‚РµСЃС‚РѕРІРѕРіРѕ Р·РѕР±СЂР°Р¶РµРЅРЅСЏ')
    parser.add_argument('--model', type=str, default='resnet50',
                        help='РќР°Р·РІР° РјРѕРґРµР»С–')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], default=None,
                        help='РџСЂРёСЃС‚СЂС–Р№ РґР»СЏ РІРёРєРѕРЅР°РЅРЅСЏ (cuda Р°Р±Рѕ cpu)')

    # РџР°СЂР°РјРµС‚СЂРё Р±РµРЅС‡РјР°СЂРєС–РЅРіСѓ
    parser.add_argument('--iterations', type=int, default=100,
                        help='РљС–Р»СЊРєС–СЃС‚СЊ С–С‚РµСЂР°С†С–Р№ РґР»СЏ РєРѕР¶РЅРѕРіРѕ С‚РµСЃС‚Сѓ')
    parser.add_argument('--component', type=str, default='all',
                        choices=['all', 'image_loading', 'preprocessing', 'model_forward', 'postprocessing', 'end_to_end', 'batch_sizes'],
                        help='РљРѕРјРїРѕРЅРµРЅС‚ РґР»СЏ Р±РµРЅС‡РјР°СЂРєС–РЅРіСѓ')
    parser.add_argument('--batch-sizes', type=str, default='1,2,4,8,16,32',
                        help='Р РѕР·РјС–СЂРё Р±Р°С‚С‡С–РІ РґР»СЏ С‚РµСЃС‚СѓРІР°РЅРЅСЏ (С‡РµСЂРµР· РєРѕРјСѓ)')

    # РџР°СЂР°РјРµС‚СЂРё РІРёС…С–РґРЅРёС… РґР°РЅРёС…
    parser.add_argument('--output-json', type=str, default=None,
                        help='РЁР»СЏС… РґР»СЏ Р·Р±РµСЂРµР¶РµРЅРЅСЏ СЂРµР·СѓР»СЊС‚Р°С‚С–РІ Сѓ JSON')
    parser.add_argument('--output-plot', type=str, default=None,
                        help='РЁР»СЏС… РґР»СЏ Р·Р±РµСЂРµР¶РµРЅРЅСЏ РіСЂР°С„С–РєС–РІ')

    args = parser.parse_args()

    # РџРµСЂРµРІС–СЂРєР° РЅР°СЏРІРЅРѕСЃС‚С– С„Р°Р№Р»Сѓ Р·РѕР±СЂР°Р¶РµРЅРЅСЏ
    if not os.path.isfile(args.image):
        print(f"РџРѕРјРёР»РєР°: С„Р°Р№Р» {args.image} РЅРµ С–СЃРЅСѓС”")
        return 1

    # РЎС‚РІРѕСЂРµРЅРЅСЏ Р±РµРЅС‡РјР°СЂРєРµСЂР°
    benchmarker = ComponentBenchmarker(model_name=args.model, device=args.device)

    # Р—Р°РїСѓСЃРє Р±РµРЅС‡РјР°СЂРєС–РЅРіСѓ
    results = {}

    if args.component == 'all':
        results = benchmarker.run_all_benchmarks(args.image, args.iterations)

        # Р’РёРІРµРґРµРЅРЅСЏ СЂРµР·СѓР»СЊС‚Р°С‚С–РІ
        for component, stats in results.items():
            print_stats(stats)

        # РџРѕР±СѓРґРѕРІР° РіСЂР°С„С–РєС–РІ
        if args.output_plot:
            output_file = args.output_plot
        else:
            output_file = None

        plot_component_comparison(results, output_file)

    elif args.component == 'batch_sizes':
        # РџР°СЂСЃРёРЅРі СЂРѕР·РјС–СЂС–РІ Р±Р°С‚С‡С–РІ
        batch_sizes = list(map(int, args.batch_sizes.split(',')))

        # Р—Р°РїСѓСЃРє Р±РµРЅС‡РјР°СЂРєС–РЅРіСѓ
        batch_results = benchmarker.benchmark_batch_sizes(args.image, batch_sizes, args.iterations)
        results['batch_sizes'] = batch_results

        # Р’РёРІРµРґРµРЅРЅСЏ СЂРµР·СѓР»СЊС‚Р°С‚С–РІ
        for stats in batch_results:
            print_stats(stats)

        # РџРѕР±СѓРґРѕРІР° РіСЂР°С„С–РєС–РІ
        if args.output_plot:
            output_file = args.output_plot
        else:
            output_file = None

        plot_batch_size_comparison(batch_results, output_file)

    else:
        # Р—Р°РїСѓСЃРє РєРѕРЅРєСЂРµС‚РЅРѕРіРѕ РєРѕРјРїРѕРЅРµРЅС‚Р°
        benchmark_method = getattr(benchmarker, f"benchmark_{args.component}")
        component_results = benchmark_method(args.image, args.iterations)
        results[args.component] = component_results

        # Р’РёРІРµРґРµРЅРЅСЏ СЂРµР·СѓР»СЊС‚Р°С‚С–РІ
        print_stats(component_results)

    # Р—Р±РµСЂРµР¶РµРЅРЅСЏ СЂРµР·СѓР»СЊС‚Р°С‚С–РІ
    if args.output_json:
        save_results_json(results, args.output_json)

    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())

