# новлена версія для PR
#!/usr/bin/env python

'''
Р‘РµРЅС‡РјР°СЂРєС–РЅРі РјРѕРґРµР»РµР№ РјР°С€РёРЅРЅРѕРіРѕ РЅР°РІС‡Р°РЅРЅСЏ РїС–СЃР»СЏ РѕРїС‚РёРјС–Р·Р°С†С–С—
'''

import os
import time
import json
import logging
import argparse
from datetime import datetime
from tabulate import tabulate
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights

# РќР°Р»Р°С€С‚СѓРІР°РЅРЅСЏ Р»РѕРіСѓРІР°РЅРЅСЏ
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('benchmark_models')

class ModelBenchmark:
    '''
    РљР»Р°СЃ РґР»СЏ Р±РµРЅС‡РјР°СЂРєС–РЅРіСѓ РјРѕРґРµР»РµР№ РјР°С€РёРЅРЅРѕРіРѕ РЅР°РІС‡Р°РЅРЅСЏ
    '''
    def __init__(self, models_dict, dataset_path=None, batch_size=32, num_runs=100, save_dir='benchmark_results'):
        '''
        Р†РЅС–С†С–Р°Р»С–Р·Р°С†С–СЏ Р±РµРЅС‡РјР°СЂРєРµСЂР°

        РџР°СЂР°РјРµС‚СЂРё:
        -----------
        models_dict: СЃР»РѕРІРЅРёРє Р· РјРѕРґРµР»СЏРјРё РґР»СЏ Р±РµРЅС‡РјР°СЂРєС–РЅРіСѓ {"name": model}
        dataset_path: С€Р»СЏС… РґРѕ РґР°С‚Р°СЃРµС‚Сѓ РґР»СЏ С‚РµСЃС‚СѓРІР°РЅРЅСЏ
        batch_size: СЂРѕР·РјС–СЂ Р±Р°С‚С‡Сѓ РґР»СЏ С–РЅС„РµСЂРµРЅСЃСѓ
        num_runs: РєС–Р»СЊРєС–СЃС‚СЊ Р·Р°РїСѓСЃРєС–РІ РґР»СЏ РІРёРјС–СЂСЋРІР°РЅРЅСЏ
        save_dir: РґРёСЂРµРєС‚РѕСЂС–СЏ РґР»СЏ Р·Р±РµСЂРµР¶РµРЅРЅСЏ СЂРµР·СѓР»СЊС‚Р°С‚С–РІ
        '''
        self.models_dict = models_dict
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.num_runs = num_runs
        self.save_dir = save_dir

        # РЎС‚РІРѕСЂРµРЅРЅСЏ РґРёСЂРµРєС‚РѕСЂС–С— РґР»СЏ СЂРµР·СѓР»СЊС‚Р°С‚С–РІ, СЏРєС‰Рѕ С—С— РЅРµРјР°С”
        os.makedirs(save_dir, exist_ok=True)

        # Р РµР·СѓР»СЊС‚Р°С‚Рё Р±РµРЅС‡РјР°СЂРєС–РЅРіСѓ
        self.results = {}

        # Р—Р°РІР°РЅС‚Р°Р¶РµРЅРЅСЏ РґР°РЅРёС… РґР»СЏ С‚РµСЃС‚СѓРІР°РЅРЅСЏ
        self.test_loader = self._load_test_data()

    def _load_test_data(self):
        '''
        Р—Р°РІР°РЅС‚Р°Р¶РµРЅРЅСЏ РґР°РЅРёС… РґР»СЏ С‚РµСЃС‚СѓРІР°РЅРЅСЏ

        РџРѕРІРµСЂС‚Р°С”:
        -----------
        Р·Р°РІР°РЅС‚Р°Р¶СѓРІР°С‡ РґР°РЅРёС…
        '''
        # РўСЂР°РЅСЃС„РѕСЂРјР°С†С–С— РґР»СЏ С‚РµСЃС‚РѕРІРѕРіРѕ РЅР°Р±РѕСЂСѓ
        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # РЎРїСЂРѕР±Р° Р·Р°РІР°РЅС‚Р°Р¶РёС‚Рё РґР°С‚Р°СЃРµС‚
        try:
            if self.dataset_path and os.path.exists(self.dataset_path):
                test_dataset = torchvision.datasets.ImageFolder(
                    self.dataset_path,
                    transform=test_transform
                )
            else:
                # Р’РёРєРѕСЂРёСЃС‚Р°РЅРЅСЏ CIFAR-10 СЏРє Р·Р°РїР°СЃРЅРѕРіРѕ РІР°СЂС–Р°РЅС‚Сѓ
                logger.info("Р’РёРєРѕСЂРёСЃС‚Р°РЅРЅСЏ CIFAR-10 СЏРє С‚РµСЃС‚РѕРІРѕРіРѕ РґР°С‚Р°СЃРµС‚Сѓ")
                test_dataset = torchvision.datasets.CIFAR10(
                    root='./data',
                    train=False,
                    download=True,
                    transform=test_transform
                )
        except Exception as e:
            logger.warning(f"РџРѕРјРёР»РєР° Р·Р°РІР°РЅС‚Р°Р¶РµРЅРЅСЏ РґР°РЅРёС…: {e}. Р’РёРєРѕСЂРёСЃС‚Р°РЅРЅСЏ CIFAR-10.")
            test_dataset = torchvision.datasets.CIFAR10(
                root='./data',
                train=False,
                download=True,
                transform=test_transform
            )

        # РЎС‚РІРѕСЂРµРЅРЅСЏ Р·Р°РІР°РЅС‚Р°Р¶СѓРІР°С‡Р° РґР°РЅРёС…
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

        return test_loader

    def _measure_model_size(self, model):
        '''
        Р’РёРјС–СЂСЋРІР°РЅРЅСЏ СЂРѕР·РјС–СЂСѓ РјРѕРґРµР»С–

        РџР°СЂР°РјРµС‚СЂРё:
        -----------
        model: РјРѕРґРµР»СЊ РґР»СЏ РІРёРјС–СЂСЋРІР°РЅРЅСЏ

        РџРѕРІРµСЂС‚Р°С”:
        -----------
        СЂРѕР·РјС–СЂ РјРѕРґРµР»С– РІ РњР‘
        '''
        # Р—Р±РµСЂРµР¶РµРЅРЅСЏ РјРѕРґРµР»С– Сѓ С‚РёРјС‡Р°СЃРѕРІРёР№ С„Р°Р№Р»
        temp_file = os.path.join(self.save_dir, "temp_model.pt")
        torch.save(model.state_dict(), temp_file)

        # РћС‚СЂРёРјР°РЅРЅСЏ СЂРѕР·РјС–СЂСѓ С„Р°Р№Р»Сѓ
        size_mb = os.path.getsize(temp_file) / (1024 * 1024)

        # Р’РёРґР°Р»РµРЅРЅСЏ С‚РёРјС‡Р°СЃРѕРІРѕРіРѕ С„Р°Р№Р»Сѓ
        os.remove(temp_file)

        return size_mb

    def _count_parameters(self, model):
        '''
        РџС–РґСЂР°С…СѓРЅРѕРє РєС–Р»СЊРєРѕСЃС‚С– РїР°СЂР°РјРµС‚СЂС–РІ РјРѕРґРµР»С–

        РџР°СЂР°РјРµС‚СЂРё:
        -----------
        model: РјРѕРґРµР»СЊ РґР»СЏ РїС–РґСЂР°С…СѓРЅРєСѓ

        РџРѕРІРµСЂС‚Р°С”:
        -----------
        РєС–Р»СЊРєС–СЃС‚СЊ РїР°СЂР°РјРµС‚СЂС–РІ
        '''
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def _evaluate_accuracy(self, model, device):
        '''
        РћС†С–РЅРєР° С‚РѕС‡РЅРѕСЃС‚С– РјРѕРґРµР»С–

        РџР°СЂР°РјРµС‚СЂРё:
        -----------
        model: РјРѕРґРµР»СЊ РґР»СЏ РѕС†С–РЅРєРё
        device: РїСЂРёСЃС‚СЂС–Р№ РґР»СЏ РѕР±С‡РёСЃР»РµРЅСЊ

        РџРѕРІРµСЂС‚Р°С”:
        -----------
        С‚РѕС‡РЅС–СЃС‚СЊ РјРѕРґРµР»С– Сѓ РІС–РґСЃРѕС‚РєР°С…
        '''
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in tqdm(self.test_loader, desc="РћС†С–РЅРєР° С‚РѕС‡РЅРѕСЃС‚С–"):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        return accuracy

    def _measure_inference_time(self, model, device):
        '''
        Р’РёРјС–СЂСЋРІР°РЅРЅСЏ С‡Р°СЃСѓ С–РЅС„РµСЂРµРЅСЃСѓ РјРѕРґРµР»С–

        РџР°СЂР°РјРµС‚СЂРё:
        -----------
        model: РјРѕРґРµР»СЊ РґР»СЏ РІРёРјС–СЂСЋРІР°РЅРЅСЏ
        device: РїСЂРёСЃС‚СЂС–Р№ РґР»СЏ РѕР±С‡РёСЃР»РµРЅСЊ

        РџРѕРІРµСЂС‚Р°С”:
        -----------
        СЃР»РѕРІРЅРёРє Р· С‡Р°СЃРѕРј С–РЅС„РµСЂРµРЅСЃСѓ (СЃРµСЂРµРґРЅС–Р№, РјС–РЅ, РјР°РєСЃ, СЃС‚Рґ)
        '''
        model.eval()
        times = []
        batch_times = []

        # РћС‚СЂРёРјР°РЅРЅСЏ РїРµСЂС€РѕРіРѕ Р±Р°С‚С‡Сѓ РґР»СЏ РІРёРјС–СЂСЋРІР°РЅРЅСЏ РѕРґРёРЅРѕС‡РЅРѕРіРѕ С–РЅС„РµСЂРµРЅСЃСѓ
        single_input = None
        for inputs, _ in self.test_loader:
            single_input = inputs[0:1].to(device)  # РћРґРёРЅ Р·СЂР°Р·РѕРє
            inputs = inputs.to(device)             # РџРѕРІРЅРёР№ Р±Р°С‚С‡
            break

        # Р РѕР·С–РіСЂС–РІ
        with torch.no_grad():
            for _ in range(10):
                _ = model(single_input)
                _ = model(inputs)

        # Р’РёРјС–СЂСЋРІР°РЅРЅСЏ С‡Р°СЃСѓ С–РЅС„РµСЂРµРЅСЃСѓ РѕРґРёРЅРѕС‡РЅРѕРіРѕ Р·СЂР°Р·РєР°
        with torch.no_grad():
            for _ in range(self.num_runs):
                start_time = time.time()
                _ = model(single_input)
                end_time = time.time()
                times.append(end_time - start_time)

        # Р’РёРјС–СЂСЋРІР°РЅРЅСЏ С‡Р°СЃСѓ С–РЅС„РµСЂРµРЅСЃСѓ Р±Р°С‚С‡Сѓ
        with torch.no_grad():
            for _ in range(self.num_runs // 10):  # РњРµРЅС€Рµ Р·Р°РїСѓСЃРєС–РІ РґР»СЏ Р±Р°С‚С‡Сѓ
                start_time = time.time()
                _ = model(inputs)
                end_time = time.time()
                batch_times.append(end_time - start_time)

        # РћР±С‡РёСЃР»РµРЅРЅСЏ СЃС‚Р°С‚РёСЃС‚РёРєРё
        time_stats = {
            "single": {
                "mean": float(np.mean(times) * 1000),  # РјСЃ
                "min": float(np.min(times) * 1000),    # РјСЃ
                "max": float(np.max(times) * 1000),    # РјСЃ
                "std": float(np.std(times) * 1000)     # РјСЃ
            },
            "batch": {
                "mean": float(np.mean(batch_times) * 1000),  # РјСЃ
                "min": float(np.min(batch_times) * 1000),    # РјСЃ
                "max": float(np.max(batch_times) * 1000),    # РјСЃ
                "std": float(np.std(batch_times) * 1000),    # РјСЃ
                "batch_size": self.batch_size
            }
        }

        return time_stats

    def _measure_memory_usage(self, model, device):
        '''
        Р’РёРјС–СЂСЋРІР°РЅРЅСЏ РІРёРєРѕСЂРёСЃС‚Р°РЅРЅСЏ РїР°Рј'СЏС‚С– РјРѕРґРµР»Р»СЋ

        РџР°СЂР°РјРµС‚СЂРё:
        -----------
        model: РјРѕРґРµР»СЊ РґР»СЏ РІРёРјС–СЂСЋРІР°РЅРЅСЏ
        device: РїСЂРёСЃС‚СЂС–Р№ РґР»СЏ РѕР±С‡РёСЃР»РµРЅСЊ

        РџРѕРІРµСЂС‚Р°С”:
        -----------
        РІРёРєРѕСЂРёСЃС‚Р°РЅРЅСЏ РїР°Рј'СЏС‚С– РІ РњР‘
        '''
        # РћС‡РёС‰РµРЅРЅСЏ РєРµС€Сѓ CUDA
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        # РћС‚СЂРёРјР°РЅРЅСЏ Р±Р°С‚С‡Сѓ РґР»СЏ С–РЅС„РµСЂРµРЅСЃСѓ
        for inputs, _ in self.test_loader:
            inputs = inputs.to(device)
            break

        # Р†РЅС„РµСЂРµРЅСЃ РґР»СЏ РІРёРјС–СЂСЋРІР°РЅРЅСЏ РїР°Рј'СЏС‚С–
        with torch.no_grad():
            _ = model(inputs)

        # Р’РёРјС–СЂСЋРІР°РЅРЅСЏ РІРёРєРѕСЂРёСЃС‚Р°РЅРЅСЏ РїР°Рј'СЏС‚С–
        if device.type == 'cuda':
            memory_usage = torch.cuda.max_memory_allocated() / (1024 * 1024)  # РњР‘
        else:
            # РџСЂРёР±Р»РёР·РЅР° РѕС†С–РЅРєР° РґР»СЏ CPU (РЅРµ С‚РѕС‡РЅР°)
            memory_usage = self._measure_model_size(model) * 2

        return float(memory_usage)

    def _profile_model(self, model, device):
        '''
        РџСЂРѕС„С–Р»СЋРІР°РЅРЅСЏ РјРѕРґРµР»С– Р·Р° РґРѕРїРѕРјРѕРіРѕСЋ torch.profiler

        РџР°СЂР°РјРµС‚СЂРё:
        -----------
        model: РјРѕРґРµР»СЊ РґР»СЏ РїСЂРѕС„С–Р»СЋРІР°РЅРЅСЏ
        device: РїСЂРёСЃС‚СЂС–Р№ РґР»СЏ РѕР±С‡РёСЃР»РµРЅСЊ

        РџРѕРІРµСЂС‚Р°С”:
        -----------
        СЃР»РѕРІРЅРёРє Р· СЂРµР·СѓР»СЊС‚Р°С‚Р°РјРё РїСЂРѕС„С–Р»СЋРІР°РЅРЅСЏ
        '''
        try:
            from torch.profiler import profile, record_function, ProfilerActivity

            # РћС‚СЂРёРјР°РЅРЅСЏ Р±Р°С‚С‡Сѓ РґР»СЏ С–РЅС„РµСЂРµРЅСЃСѓ
            for inputs, _ in self.test_loader:
                inputs = inputs.to(device)
                break

            # Р РѕР·С–РіСЂС–РІ
            with torch.no_grad():
                for _ in range(5):
                    _ = model(inputs)

            # РџСЂРѕС„С–Р»СЋРІР°РЅРЅСЏ
            activities = [ProfilerActivity.CPU]
            if device.type == 'cuda':
                activities.append(ProfilerActivity.CUDA)

            with torch.no_grad():
                with profile(activities=activities, record_shapes=True) as prof:
                    with record_function("model_inference"):
                        _ = model(inputs)

            # РђРЅР°Р»С–Р· СЂРµР·СѓР»СЊС‚Р°С‚С–РІ РїСЂРѕС„С–Р»СЋРІР°РЅРЅСЏ
            total_time = prof.self_cpu_time_total / 1000.0  # РјСЃ
            table = prof.key_averages().table(sort_by="cpu_time_total", row_limit=10)

            # РћС‚СЂРёРјР°РЅРЅСЏ С‚РѕРї-5 РЅР°Р№РґРѕРІС€РёС… РѕРїРµСЂР°С†С–Р№
            top_ops = []
            for row in prof.key_averages().table(sort_by="cpu_time_total", row_limit=5).split("\n")[2:-1]:
                parts = row.split()
                if len(parts) >= 7:
                    op_name = parts[6]
                    cpu_time = float(parts[1])
                    top_ops.append({"name": op_name, "cpu_time": cpu_time})

            return {
                "total_time": float(total_time),
                "top_ops": top_ops,
                "profile_table": table
            }

        except Exception as e:
            logger.warning(f"РџРѕРјРёР»РєР° РїСЂРѕС„С–Р»СЋРІР°РЅРЅСЏ: {e}")
            return {"error": str(e)}

    def benchmark_model(self, model_name):
        '''
        Р‘РµРЅС‡РјР°СЂРєС–РЅРі РјРѕРґРµР»С–

        РџР°СЂР°РјРµС‚СЂРё:
        -----------
        model_name: РЅР°Р·РІР° РјРѕРґРµР»С– Сѓ СЃР»РѕРІРЅРёРєСѓ РјРѕРґРµР»РµР№

        РџРѕРІРµСЂС‚Р°С”:
        -----------
        СЃР»РѕРІРЅРёРє Р· СЂРµР·СѓР»СЊС‚Р°С‚Р°РјРё Р±РµРЅС‡РјР°СЂРєС–РЅРіСѓ
        '''
        if model_name not in self.models_dict:
            logger.error(f"РњРѕРґРµР»СЊ {model_name} РЅРµ Р·РЅР°Р№РґРµРЅР° Сѓ СЃР»РѕРІРЅРёРєСѓ РјРѕРґРµР»РµР№")
            return None

        model = self.models_dict[model_name]
        logger.info(f"Р‘РµРЅС‡РјР°СЂРєС–РЅРі РјРѕРґРµР»С–: {model_name}")

        # Р’РёР·РЅР°С‡РµРЅРЅСЏ РїСЂРёСЃС‚СЂРѕСЋ
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        # Р’РёРјС–СЂСЋРІР°РЅРЅСЏ РѕСЃРЅРѕРІРЅРёС… РјРµС‚СЂРёРє
        model_size = self._measure_model_size(model)
        logger.info(f"Р РѕР·РјС–СЂ РјРѕРґРµР»С–: {model_size:.2f} РњР‘")

        num_params = self._count_parameters(model)
        logger.info(f"РљС–Р»СЊРєС–СЃС‚СЊ РїР°СЂР°РјРµС‚СЂС–РІ: {num_params:,}")

        memory_usage = self._measure_memory_usage(model, device)
        logger.info(f"Р’РёРєРѕСЂРёСЃС‚Р°РЅРЅСЏ РїР°Рј'СЏС‚С–: {memory_usage:.2f} РњР‘")

        accuracy = self._evaluate_accuracy(model, device)
        logger.info(f"РўРѕС‡РЅС–СЃС‚СЊ: {accuracy:.2f}%")

        inference_time = self._measure_inference_time(model, device)
        logger.info(f"РЎРµСЂРµРґРЅС–Р№ С‡Р°СЃ С–РЅС„РµСЂРµРЅСЃСѓ (РѕРґРёРЅРѕС‡РЅРёР№): {inference_time['single']['mean']:.2f} РјСЃ")
        logger.info(f"РЎРµСЂРµРґРЅС–Р№ С‡Р°СЃ С–РЅС„РµСЂРµРЅСЃСѓ (Р±Р°С‚С‡): {inference_time['batch']['mean']:.2f} РјСЃ")

        # РџСЂРѕС„С–Р»СЋРІР°РЅРЅСЏ РјРѕРґРµР»С–
        profile_results = self._profile_model(model, device)

        # Р—Р±РµСЂРµР¶РµРЅРЅСЏ СЂРµР·СѓР»СЊС‚Р°С‚С–РІ
        result = {
            "name": model_name,
            "device": device.type,
            "size_mb": model_size,
            "parameters": num_params,
            "memory_usage_mb": memory_usage,
            "accuracy": accuracy,
            "inference_time": inference_time,
            "throughput_samples_per_sec": self.batch_size / (inference_time["batch"]["mean"] / 1000),
            "profile": profile_results,
            "timestamp": datetime.now().isoformat()
        }

        self.results[model_name] = result
        return result

    def run_all_benchmarks(self):
        '''
        Р—Р°РїСѓСЃРє Р±РµРЅС‡РјР°СЂРєС–РЅРіСѓ РґР»СЏ РІСЃС–С… РјРѕРґРµР»РµР№

        РџРѕРІРµСЂС‚Р°С”:
        -----------
        СЃР»РѕРІРЅРёРє Р· СЂРµР·СѓР»СЊС‚Р°С‚Р°РјРё Р±РµРЅС‡РјР°СЂРєС–РЅРіСѓ
        '''
        for model_name in self.models_dict.keys():
            self.benchmark_model(model_name)

        # Р—Р±РµСЂРµР¶РµРЅРЅСЏ СЂРµР·СѓР»СЊС‚Р°С‚С–РІ
        self.save_results()

        return self.results

    def save_results(self, filename=None):
        '''
        Р—Р±РµСЂРµР¶РµРЅРЅСЏ СЂРµР·СѓР»СЊС‚Р°С‚С–РІ Р±РµРЅС‡РјР°СЂРєС–РЅРіСѓ

        РџР°СЂР°РјРµС‚СЂРё:
        -----------
        filename: С–Рј'СЏ С„Р°Р№Р»Сѓ РґР»СЏ Р·Р±РµСЂРµР¶РµРЅРЅСЏ (СЏРєС‰Рѕ None, РіРµРЅРµСЂСѓС”С‚СЊСЃСЏ Р°РІС‚РѕРјР°С‚РёС‡РЅРѕ)
        '''
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_results_{timestamp}.json"

        file_path = os.path.join(self.save_dir, filename)

        with open(file_path, 'w') as f:
            json.dump(self.results, f, indent=2)

        logger.info(f"Р РµР·СѓР»СЊС‚Р°С‚Рё Р·Р±РµСЂРµР¶РµРЅРѕ Сѓ {file_path}")

    def generate_report(self, output_format='text'):
        '''
        Р“РµРЅРµСЂР°С†С–СЏ Р·РІС–С‚Сѓ Р· СЂРµР·СѓР»СЊС‚Р°С‚Р°РјРё Р±РµРЅС‡РјР°СЂРєС–РЅРіСѓ

        РџР°СЂР°РјРµС‚СЂРё:
        -----------
        output_format: С„РѕСЂРјР°С‚ Р·РІС–С‚Сѓ ('text', 'html', 'markdown')

        РџРѕРІРµСЂС‚Р°С”:
        -----------
        Р·РІС–С‚ Сѓ РІРєР°Р·Р°РЅРѕРјСѓ С„РѕСЂРјР°С‚С–
        '''
        if not self.results:
            logger.warning("РќРµРјР°С” СЂРµР·СѓР»СЊС‚Р°С‚С–РІ РґР»СЏ РіРµРЅРµСЂР°С†С–С— Р·РІС–С‚Сѓ")
            return "РќРµРјР°С” СЂРµР·СѓР»СЊС‚Р°С‚С–РІ РґР»СЏ РіРµРЅРµСЂР°С†С–С— Р·РІС–С‚Сѓ"

        # РџС–РґРіРѕС‚РѕРІРєР° РґР°РЅРёС… РґР»СЏ С‚Р°Р±Р»РёС†С–
        headers = ["РњРѕРґРµР»СЊ", "Р РѕР·РјС–СЂ (РњР‘)", "РџР°СЂР°РјРµС‚СЂРё", "РџР°Рј'СЏС‚СЊ (РњР‘)", "РўРѕС‡РЅС–СЃС‚СЊ (%)", "Р†РЅС„РµСЂРµРЅСЃ (РјСЃ)", "РџСЂРѕРїСѓСЃРєРЅР° Р·РґР°С‚РЅС–СЃС‚СЊ (Р·СЂР°Р·РєС–РІ/СЃ)"]
        rows = []

        for model_name, result in self.results.items():
            rows.append([
                model_name,
                f"{result['size_mb']:.2f}",
                f"{result['parameters']:,}",
                f"{result['memory_usage_mb']:.2f}",
                f"{result['accuracy']:.2f}",
                f"{result['inference_time']['single']['mean']:.2f}",
                f"{result['throughput_samples_per_sec']:.2f}"
            ])

        # Р“РµРЅРµСЂР°С†С–СЏ С‚Р°Р±Р»РёС†С– Сѓ РІРєР°Р·Р°РЅРѕРјСѓ С„РѕСЂРјР°С‚С–
        if output_format == 'html':
            table = tabulate(rows, headers, tablefmt="html")
            report = f"<h1>Р—РІС–С‚ Р· Р±РµРЅС‡РјР°СЂРєС–РЅРіСѓ РјРѕРґРµР»РµР№</h1>\n<p>Р”Р°С‚Р°: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>\n{table}"
        elif output_format == 'markdown':
            table = tabulate(rows, headers, tablefmt="pipe")
            report = f"# Р—РІС–С‚ Р· Р±РµРЅС‡РјР°СЂРєС–РЅРіСѓ РјРѕРґРµР»РµР№\nР”Р°С‚Р°: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n{table}"
        else:  # text
            table = tabulate(rows, headers, tablefmt="grid")
            report = f"Р—РІС–С‚ Р· Р±РµРЅС‡РјР°СЂРєС–РЅРіСѓ РјРѕРґРµР»РµР№\nР”Р°С‚Р°: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n{table}"

        return report

    def generate_plots(self):
        '''
        Р“РµРЅРµСЂР°С†С–СЏ РіСЂР°С„С–РєС–РІ Р· СЂРµР·СѓР»СЊС‚Р°С‚Р°РјРё Р±РµРЅС‡РјР°СЂРєС–РЅРіСѓ
        '''
        if not self.results:
            logger.warning("РќРµРјР°С” СЂРµР·СѓР»СЊС‚Р°С‚С–РІ РґР»СЏ РіРµРЅРµСЂР°С†С–С— РіСЂР°С„С–РєС–РІ")
            return

        # РџС–РґРіРѕС‚РѕРІРєР° РґР°РЅРёС… РґР»СЏ РіСЂР°С„С–РєС–РІ
        model_names = list(self.results.keys())
        sizes = [self.results[m]['size_mb'] for m in model_names]
        accuracies = [self.results[m]['accuracy'] for m in model_names]
        inference_times = [self.results[m]['inference_time']['single']['mean'] for m in model_names]
        throughputs = [self.results[m]['throughput_samples_per_sec'] for m in model_names]

        # РЎС‚РІРѕСЂРµРЅРЅСЏ РіСЂР°С„С–РєС–РІ
        plt.figure(figsize=(15, 10))

        # Р“СЂР°С„С–Рє СЂРѕР·РјС–СЂС–РІ РјРѕРґРµР»РµР№
        plt.subplot(2, 2, 1)
        plt.bar(model_names, sizes, color='skyblue')
        plt.title('Р РѕР·РјС–СЂ РјРѕРґРµР»С– (РњР‘)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        # Р“СЂР°С„С–Рє С‚РѕС‡РЅРѕСЃС‚С–
        plt.subplot(2, 2, 2)
        plt.bar(model_names, accuracies, color='lightgreen')
        plt.title('РўРѕС‡РЅС–СЃС‚СЊ (%)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        # Р“СЂР°С„С–Рє С‡Р°СЃСѓ С–РЅС„РµСЂРµРЅСЃСѓ
        plt.subplot(2, 2, 3)
        plt.bar(model_names, inference_times, color='salmon')
        plt.title('Р§Р°СЃ С–РЅС„РµСЂРµРЅСЃСѓ (РјСЃ)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        # Р“СЂР°С„С–Рє РїСЂРѕРїСѓСЃРєРЅРѕС— Р·РґР°С‚РЅРѕСЃС‚С–
        plt.subplot(2, 2, 4)
        plt.bar(model_names, throughputs, color='mediumpurple')
        plt.title('РџСЂРѕРїСѓСЃРєРЅР° Р·РґР°С‚РЅС–СЃС‚СЊ (Р·СЂР°Р·РєС–РІ/СЃ)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        # Р—Р±РµСЂРµР¶РµРЅРЅСЏ РіСЂР°С„С–РєС–РІ
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = os.path.join(self.save_dir, f"benchmark_plots_{timestamp}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"Р“СЂР°С„С–РєРё Р·Р±РµСЂРµР¶РµРЅРѕ Сѓ {plot_path}")

        # Р”РѕРґР°С‚РєРѕРІРёР№ РіСЂР°С„С–Рє: СЃРїС–РІРІС–РґРЅРѕС€РµРЅРЅСЏ С‚РѕС‡РЅРѕСЃС‚С– С‚Р° С‡Р°СЃСѓ С–РЅС„РµСЂРµРЅСЃСѓ
        plt.figure(figsize=(10, 6))
        plt.scatter(inference_times, accuracies, s=100, alpha=0.7)

        # Р”РѕРґР°РІР°РЅРЅСЏ РїС–РґРїРёСЃС–РІ РґРѕ С‚РѕС‡РѕРє
        for i, model_name in enumerate(model_names):
            plt.annotate(model_name, (inference_times[i], accuracies[i]), 
                        textcoords="offset points", xytext=(0,10), ha='center')

        plt.xlabel('Р§Р°СЃ С–РЅС„РµСЂРµРЅСЃСѓ (РјСЃ)')
        plt.ylabel('РўРѕС‡РЅС–СЃС‚СЊ (%)')
        plt.title('РЎРїС–РІРІС–РґРЅРѕС€РµРЅРЅСЏ С‚РѕС‡РЅРѕСЃС‚С– С‚Р° С‡Р°СЃСѓ С–РЅС„РµСЂРµРЅСЃСѓ')
        plt.grid(True, linestyle='--', alpha=0.7)

        # Р—Р±РµСЂРµР¶РµРЅРЅСЏ РіСЂР°С„С–РєСѓ
        scatter_path = os.path.join(self.save_dir, f"accuracy_vs_speed_{timestamp}.png")
        plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
        logger.info(f"Р“СЂР°С„С–Рє СЃРїС–РІРІС–РґРЅРѕС€РµРЅРЅСЏ Р·Р±РµСЂРµР¶РµРЅРѕ Сѓ {scatter_path}")

def load_models(model_paths):
    '''
    Р—Р°РІР°РЅС‚Р°Р¶РµРЅРЅСЏ РјРѕРґРµР»РµР№ Р· С„Р°Р№Р»С–РІ

    РџР°СЂР°РјРµС‚СЂРё:
    -----------
    model_paths: СЃР»РѕРІРЅРёРє {"РЅР°Р·РІР°": "С€Р»СЏС…"}

    РџРѕРІРµСЂС‚Р°С”:
    -----------
    СЃР»РѕРІРЅРёРє Р· РјРѕРґРµР»СЏРјРё {"РЅР°Р·РІР°": РјРѕРґРµР»СЊ}
    '''
    models_dict = {}

    for name, path in model_paths.items():
        try:
            logger.info(f"Р—Р°РІР°РЅС‚Р°Р¶РµРЅРЅСЏ РјРѕРґРµР»С– {name} Р· {path}")
            model = torch.load(path, map_location=torch.device('cpu'))
            model.eval()
            models_dict[name] = model
        except Exception as e:
            logger.error(f"РџРѕРјРёР»РєР° Р·Р°РІР°РЅС‚Р°Р¶РµРЅРЅСЏ РјРѕРґРµР»С– {name}: {e}")

    return models_dict

def main():
    parser = argparse.ArgumentParser(description="Р‘РµРЅС‡РјР°СЂРєС–РЅРі РјРѕРґРµР»РµР№ РјР°С€РёРЅРЅРѕРіРѕ РЅР°РІС‡Р°РЅРЅСЏ")
    parser.add_argument("--models", type=str, required=True,
                        help="РЁР»СЏС…Рё РґРѕ РјРѕРґРµР»РµР№ Сѓ С„РѕСЂРјР°С‚С– 'РЅР°Р·РІР°:С€Р»СЏС…,РЅР°Р·РІР°2:С€Р»СЏС…2'")
    parser.add_argument("--dataset", type=str, default=None,
                        help="РЁР»СЏС… РґРѕ С‚РµСЃС‚РѕРІРѕРіРѕ РґР°С‚Р°СЃРµС‚Сѓ")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Р РѕР·РјС–СЂ Р±Р°С‚С‡Сѓ РґР»СЏ С–РЅС„РµСЂРµРЅСЃСѓ")
    parser.add_argument("--num-runs", type=int, default=100,
                        help="РљС–Р»СЊРєС–СЃС‚СЊ Р·Р°РїСѓСЃРєС–РІ РґР»СЏ РІРёРјС–СЂСЋРІР°РЅРЅСЏ С‡Р°СЃСѓ С–РЅС„РµСЂРµРЅСЃСѓ")
    parser.add_argument("--save-dir", type=str, default="benchmark_results",
                        help="Р”РёСЂРµРєС‚РѕСЂС–СЏ РґР»СЏ Р·Р±РµСЂРµР¶РµРЅРЅСЏ СЂРµР·СѓР»СЊС‚Р°С‚С–РІ")
    parser.add_argument("--report-format", type=str, choices=['text', 'html', 'markdown'], default="markdown",
                        help="Р¤РѕСЂРјР°С‚ Р·РІС–С‚Сѓ")

    args = parser.parse_args()

    # РџР°СЂСЃРёРЅРі С€Р»СЏС…С–РІ РґРѕ РјРѕРґРµР»РµР№
    model_paths = {}
    for model_spec in args.models.split(','):
        if ':' in model_spec:
            name, path = model_spec.split(':', 1)
            model_paths[name] = path

    if not model_paths:
        logger.error("РќРµ РІРєР°Р·Р°РЅРѕ Р¶РѕРґРЅРѕС— РјРѕРґРµР»С– РґР»СЏ Р±РµРЅС‡РјР°СЂРєС–РЅРіСѓ")
        return

    # Р—Р°РІР°РЅС‚Р°Р¶РµРЅРЅСЏ РјРѕРґРµР»РµР№
    models_dict = load_models(model_paths)

    if not models_dict:
        logger.error("РќРµ РІРґР°Р»РѕСЃСЏ Р·Р°РІР°РЅС‚Р°Р¶РёС‚Рё Р¶РѕРґРЅРѕС— РјРѕРґРµР»С–")
        return

    # РЎС‚РІРѕСЂРµРЅРЅСЏ С‚Р° Р·Р°РїСѓСЃРє Р±РµРЅС‡РјР°СЂРєРµСЂР°
    benchmarker = ModelBenchmark(
        models_dict=models_dict,
        dataset_path=args.dataset,
        batch_size=args.batch_size,
        num_runs=args.num_runs,
        save_dir=args.save_dir
    )

    # Р—Р°РїСѓСЃРє Р±РµРЅС‡РјР°СЂРєС–РЅРіСѓ
    results = benchmarker.run_all_benchmarks()

    # Р“РµРЅРµСЂР°С†С–СЏ Р·РІС–С‚Сѓ
    report = benchmarker.generate_report(output_format=args.report_format)
    report_path = os.path.join(args.save_dir, f"benchmark_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{args.report_format}")

    with open(report_path, 'w') as f:
        f.write(report)

    logger.info(f"Р—РІС–С‚ Р·Р±РµСЂРµР¶РµРЅРѕ Сѓ {report_path}")

    # Р“РµРЅРµСЂР°С†С–СЏ РіСЂР°С„С–РєС–РІ
    benchmarker.generate_plots()

    logger.info("Р‘РµРЅС‡РјР°СЂРєС–РЅРі Р·Р°РІРµСЂС€РµРЅРѕ")

if __name__ == "__main__":
    main()

