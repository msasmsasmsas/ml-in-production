#!/usr/bin/env python

'''
Створення flamegraph для моделей машинного навчання
'''

import os
import sys
import time
import argparse
import logging
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as transforms

# Налаштування логування
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('model_flamegraph')

class FlameGraphProfiler:
    '''
    Клас для профілювання моделей та створення flamegraph
    '''
    def __init__(self, model, dataset_path=None, batch_size=1, save_dir='profile_results'):
        '''
        Ініціалізація профілювальника

        Параметри:
        -----------
        model: модель для профілювання
        dataset_path: шлях до датасету для тестування
        batch_size: розмір батчу для інференсу
        save_dir: директорія для збереження результатів
        '''
        self.model = model
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.save_dir = save_dir

        # Створення директорії для результатів, якщо її немає
        os.makedirs(save_dir, exist_ok=True)

        # Завантаження даних для тестування
        self.test_loader = self._load_test_data()

    def _load_test_data(self):
        '''
        Завантаження даних для тестування

        Повертає:
        -----------
        завантажувач даних
        '''
        # Трансформації для тестового набору
        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # Спроба завантажити датасет
        try:
            if self.dataset_path and os.path.exists(self.dataset_path):
                test_dataset = torchvision.datasets.ImageFolder(
                    self.dataset_path,
                    transform=test_transform
                )
            else:
                # Використання CIFAR-10 як запасного варіанту
                logger.info("Використання CIFAR-10 як тестового датасету")
                test_dataset = torchvision.datasets.CIFAR10(
                    root='./data',
                    train=False,
                    download=True,
                    transform=test_transform
                )
        except Exception as e:
            logger.warning(f"Помилка завантаження даних: {e}. Використання CIFAR-10.")
            test_dataset = torchvision.datasets.CIFAR10(
                root='./data',
                train=False,
                download=True,
                transform=test_transform
            )

        # Створення завантажувача даних
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,  # Важливо для профілювання
            pin_memory=False
        )

        return test_loader

    def profile_with_torch_profiler(self, trace_path=None, use_cuda=True):
        '''
        Профілювання моделі за допомогою torch.profiler

        Параметри:
        -----------
        trace_path: шлях для збереження trace файлу
        use_cuda: використовувати CUDA, якщо доступно

        Повертає:
        -----------
        результати профілювання
        '''
        try:
            from torch.profiler import profile, record_function, ProfilerActivity, tensorboard_trace_handler

            # Вибір пристрою
            device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
            model = self.model.to(device)
            model.eval()

            # Отримання батчу для інференсу
            for inputs, _ in self.test_loader:
                inputs = inputs.to(device)
                break

            # Налаштування шляху для trace
            if trace_path is None:
                trace_path = os.path.join(self.save_dir, f"torch_trace_{int(time.time())}")

            # Розігрів
            with torch.no_grad():
                for _ in range(5):
                    _ = model(inputs)

            # Налаштування активностей для профілювання
            activities = [ProfilerActivity.CPU]
            if device.type == 'cuda':
                activities.append(ProfilerActivity.CUDA)

            # Профілювання
            logger.info(f"Профілювання моделі на {device.type}...")
            with torch.no_grad():
                with profile(
                    activities=activities,
                    record_shapes=True,
                    profile_memory=True,
                    with_stack=True,
                    on_trace_ready=tensorboard_trace_handler(trace_path)
                ) as prof:
                    with record_function("model_inference"):
                        _ = model(inputs)

            # Збереження текстового звіту
            text_path = os.path.join(self.save_dir, f"profile_report_{int(time.time())}.txt")
            with open(text_path, 'w') as f:
                f.write(prof.key_averages().table(sort_by="cpu_time_total", row_limit=50))

            logger.info(f"Звіт профілювання збережено у {text_path}")
            logger.info(f"Trace збережено у {trace_path}")
            logger.info(f"Для перегляду flamegraph запустіть: tensorboard --logdir={trace_path}")

            return prof

        except Exception as e:
            logger.error(f"Помилка профілювання: {e}")
            return None

    def profile_with_pyinstrument(self, html_path=None):
        '''
        Профілювання моделі за допомогою pyinstrument

        Параметри:
        -----------
        html_path: шлях для збереження HTML звіту

        Повертає:
        -----------
        результати профілювання
        '''
        try:
            from pyinstrument import Profiler

            # Переведення моделі на CPU (pyinstrument не підтримує CUDA)
            device = torch.device("cpu")
            model = self.model.to(device)
            model.eval()

            # Отримання батчу для інференсу
            for inputs, _ in self.test_loader:
                inputs = inputs.to(device)
                break

            # Налаштування шляху для HTML
            if html_path is None:
                html_path = os.path.join(self.save_dir, f"pyinstrument_profile_{int(time.time())}.html")

            # Розігрів
            with torch.no_grad():
                for _ in range(5):
                    _ = model(inputs)

            # Профілювання
            logger.info("Профілювання моделі з pyinstrument...")
            profiler = Profiler()
            profiler.start()

            with torch.no_grad():
                _ = model(inputs)

            profiler.stop()

            # Збереження HTML звіту
            with open(html_path, 'w') as f:
                f.write(profiler.output_html())

            logger.info(f"HTML звіт профілювання збережено у {html_path}")

            return profiler

        except ImportError:
            logger.error("Не вдалося імпортувати pyinstrument. Встановіть його за допомогою 'pip install pyinstrument'")
            return None
        except Exception as e:
            logger.error(f"Помилка профілювання з pyinstrument: {e}")
            return None

    def profile_with_cprofile(self, output_path=None):
        '''
        Профілювання моделі за допомогою cProfile

        Параметри:
        -----------
        output_path: шлях для збереження результатів профілювання

        Повертає:
        -----------
        результати профілювання
        '''
        try:
            import cProfile
            import pstats
            import io

            # Переведення моделі на CPU
            device = torch.device("cpu")
            model = self.model.to(device)
            model.eval()

            # Отримання батчу для інференсу
            for inputs, _ in self.test_loader:
                inputs = inputs.to(device)
                break

            # Налаштування шляху для результатів
            if output_path is None:
                output_path = os.path.join(self.save_dir, f"cprofile_stats_{int(time.time())}.txt")

            # Розігрів
            with torch.no_grad():
                for _ in range(5):
                    _ = model(inputs)

            # Профілювання
            logger.info("Профілювання моделі з cProfile...")
            profiler = cProfile.Profile()
            profiler.enable()

            with torch.no_grad():
                _ = model(inputs)

            profiler.disable()

            # Збереження результатів
            s = io.StringIO()
            ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
            ps.print_stats(50)  # Топ-50 функцій за кумулятивним часом

            with open(output_path, 'w') as f:
                f.write(s.getvalue())

            logger.info(f"cProfile звіт збережено у {output_path}")

            return profiler

        except Exception as e:
            logger.error(f"Помилка профілювання з cProfile: {e}")
            return None

    def run_all_profilers(self, model_name="model"):
        '''
        Запуск всіх доступних профілювальників

        Параметри:
        -----------
        model_name: назва моделі для іменування файлів

        Повертає:
        -----------
        словник з результатами профілювання
        '''
        results = {}

        # Профілювання з torch.profiler
        trace_path = os.path.join(self.save_dir, f"{model_name}_torch_trace_{int(time.time())}")
        results["torch_profiler"] = self.profile_with_torch_profiler(trace_path=trace_path)

        # Профілювання з pyinstrument
        html_path = os.path.join(self.save_dir, f"{model_name}_pyinstrument_{int(time.time())}.html")
        results["pyinstrument"] = self.profile_with_pyinstrument(html_path=html_path)

        # Профілювання з cProfile
        cprofile_path = os.path.join(self.save_dir, f"{model_name}_cprofile_{int(time.time())}.txt")
        results["cprofile"] = self.profile_with_cprofile(output_path=cprofile_path)

        return results

def main():
    parser = argparse.ArgumentParser(description="Створення flamegraph для моделей машинного навчання")
    parser.add_argument("--model", type=str, required=True,
                        help="Шлях до моделі для профілювання")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Шлях до тестового датасету")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Розмір батчу для інференсу")
    parser.add_argument("--save-dir", type=str, default="profile_results",
                        help="Директорія для збереження результатів")
    parser.add_argument("--use-cuda", action="store_true",
                        help="Використовувати CUDA, якщо доступно")
    parser.add_argument("--profiler", type=str, choices=["all", "torch", "pyinstrument", "cprofile"], default="all",
                        help="Який профілювальник використовувати")

    args = parser.parse_args()

    # Завантаження моделі
    try:
        logger.info(f"Завантаження моделі з {args.model}")
        model = torch.load(args.model, map_location=torch.device('cpu'))
        model.eval()
    except Exception as e:
        logger.error(f"Помилка завантаження моделі: {e}")
        return

    # Створення профілювальника
    profiler = FlameGraphProfiler(
        model=model,
        dataset_path=args.dataset,
        batch_size=args.batch_size,
        save_dir=args.save_dir
    )

    # Запуск профілювання
    model_name = os.path.basename(args.model).split('.')[0]

    if args.profiler == "all":
        profiler.run_all_profilers(model_name=model_name)
    elif args.profiler == "torch":
        trace_path = os.path.join(args.save_dir, f"{model_name}_torch_trace_{int(time.time())}")
        profiler.profile_with_torch_profiler(trace_path=trace_path, use_cuda=args.use_cuda)
    elif args.profiler == "pyinstrument":
        html_path = os.path.join(args.save_dir, f"{model_name}_pyinstrument_{int(time.time())}.html")
        profiler.profile_with_pyinstrument(html_path=html_path)
    elif args.profiler == "cprofile":
        cprofile_path = os.path.join(args.save_dir, f"{model_name}_cprofile_{int(time.time())}.txt")
        profiler.profile_with_cprofile(output_path=cprofile_path)

    logger.info("Профілювання завершено")

if __name__ == "__main__":
    main()
