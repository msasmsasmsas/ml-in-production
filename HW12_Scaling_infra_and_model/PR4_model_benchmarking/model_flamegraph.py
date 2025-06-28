# новлена версія для PR
#!/usr/bin/env python

'''
РЎС‚РІРѕСЂРµРЅРЅСЏ flamegraph РґР»СЏ РјРѕРґРµР»РµР№ РјР°С€РёРЅРЅРѕРіРѕ РЅР°РІС‡Р°РЅРЅСЏ
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

# РќР°Р»Р°С€С‚СѓРІР°РЅРЅСЏ Р»РѕРіСѓРІР°РЅРЅСЏ
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('model_flamegraph')

class FlameGraphProfiler:
    '''
    РљР»Р°СЃ РґР»СЏ РїСЂРѕС„С–Р»СЋРІР°РЅРЅСЏ РјРѕРґРµР»РµР№ С‚Р° СЃС‚РІРѕСЂРµРЅРЅСЏ flamegraph
    '''
    def __init__(self, model, dataset_path=None, batch_size=1, save_dir='profile_results'):
        '''
        Р†РЅС–С†С–Р°Р»С–Р·Р°С†С–СЏ РїСЂРѕС„С–Р»СЋРІР°Р»СЊРЅРёРєР°

        РџР°СЂР°РјРµС‚СЂРё:
        -----------
        model: РјРѕРґРµР»СЊ РґР»СЏ РїСЂРѕС„С–Р»СЋРІР°РЅРЅСЏ
        dataset_path: С€Р»СЏС… РґРѕ РґР°С‚Р°СЃРµС‚Сѓ РґР»СЏ С‚РµСЃС‚СѓРІР°РЅРЅСЏ
        batch_size: СЂРѕР·РјС–СЂ Р±Р°С‚С‡Сѓ РґР»СЏ С–РЅС„РµСЂРµРЅСЃСѓ
        save_dir: РґРёСЂРµРєС‚РѕСЂС–СЏ РґР»СЏ Р·Р±РµСЂРµР¶РµРЅРЅСЏ СЂРµР·СѓР»СЊС‚Р°С‚С–РІ
        '''
        self.model = model
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.save_dir = save_dir

        # РЎС‚РІРѕСЂРµРЅРЅСЏ РґРёСЂРµРєС‚РѕСЂС–С— РґР»СЏ СЂРµР·СѓР»СЊС‚Р°С‚С–РІ, СЏРєС‰Рѕ С—С— РЅРµРјР°С”
        os.makedirs(save_dir, exist_ok=True)

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
            num_workers=0,  # Р’Р°Р¶Р»РёРІРѕ РґР»СЏ РїСЂРѕС„С–Р»СЋРІР°РЅРЅСЏ
            pin_memory=False
        )

        return test_loader

    def profile_with_torch_profiler(self, trace_path=None, use_cuda=True):
        '''
        РџСЂРѕС„С–Р»СЋРІР°РЅРЅСЏ РјРѕРґРµР»С– Р·Р° РґРѕРїРѕРјРѕРіРѕСЋ torch.profiler

        РџР°СЂР°РјРµС‚СЂРё:
        -----------
        trace_path: С€Р»СЏС… РґР»СЏ Р·Р±РµСЂРµР¶РµРЅРЅСЏ trace С„Р°Р№Р»Сѓ
        use_cuda: РІРёРєРѕСЂРёСЃС‚РѕРІСѓРІР°С‚Рё CUDA, СЏРєС‰Рѕ РґРѕСЃС‚СѓРїРЅРѕ

        РџРѕРІРµСЂС‚Р°С”:
        -----------
        СЂРµР·СѓР»СЊС‚Р°С‚Рё РїСЂРѕС„С–Р»СЋРІР°РЅРЅСЏ
        '''
        try:
            from torch.profiler import profile, record_function, ProfilerActivity, tensorboard_trace_handler

            # Р’РёР±С–СЂ РїСЂРёСЃС‚СЂРѕСЋ
            device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
            model = self.model.to(device)
            model.eval()

            # РћС‚СЂРёРјР°РЅРЅСЏ Р±Р°С‚С‡Сѓ РґР»СЏ С–РЅС„РµСЂРµРЅСЃСѓ
            for inputs, _ in self.test_loader:
                inputs = inputs.to(device)
                break

            # РќР°Р»Р°С€С‚СѓРІР°РЅРЅСЏ С€Р»СЏС…Сѓ РґР»СЏ trace
            if trace_path is None:
                trace_path = os.path.join(self.save_dir, f"torch_trace_{int(time.time())}")

            # Р РѕР·С–РіСЂС–РІ
            with torch.no_grad():
                for _ in range(5):
                    _ = model(inputs)

            # РќР°Р»Р°С€С‚СѓРІР°РЅРЅСЏ Р°РєС‚РёРІРЅРѕСЃС‚РµР№ РґР»СЏ РїСЂРѕС„С–Р»СЋРІР°РЅРЅСЏ
            activities = [ProfilerActivity.CPU]
            if device.type == 'cuda':
                activities.append(ProfilerActivity.CUDA)

            # РџСЂРѕС„С–Р»СЋРІР°РЅРЅСЏ
            logger.info(f"РџСЂРѕС„С–Р»СЋРІР°РЅРЅСЏ РјРѕРґРµР»С– РЅР° {device.type}...")
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

            # Р—Р±РµСЂРµР¶РµРЅРЅСЏ С‚РµРєСЃС‚РѕРІРѕРіРѕ Р·РІС–С‚Сѓ
            text_path = os.path.join(self.save_dir, f"profile_report_{int(time.time())}.txt")
            with open(text_path, 'w') as f:
                f.write(prof.key_averages().table(sort_by="cpu_time_total", row_limit=50))

            logger.info(f"Р—РІС–С‚ РїСЂРѕС„С–Р»СЋРІР°РЅРЅСЏ Р·Р±РµСЂРµР¶РµРЅРѕ Сѓ {text_path}")
            logger.info(f"Trace Р·Р±РµСЂРµР¶РµРЅРѕ Сѓ {trace_path}")
            logger.info(f"Р”Р»СЏ РїРµСЂРµРіР»СЏРґСѓ flamegraph Р·Р°РїСѓСЃС‚С–С‚СЊ: tensorboard --logdir={trace_path}")

            return prof

        except Exception as e:
            logger.error(f"РџРѕРјРёР»РєР° РїСЂРѕС„С–Р»СЋРІР°РЅРЅСЏ: {e}")
            return None

    def profile_with_pyinstrument(self, html_path=None):
        '''
        РџСЂРѕС„С–Р»СЋРІР°РЅРЅСЏ РјРѕРґРµР»С– Р·Р° РґРѕРїРѕРјРѕРіРѕСЋ pyinstrument

        РџР°СЂР°РјРµС‚СЂРё:
        -----------
        html_path: С€Р»СЏС… РґР»СЏ Р·Р±РµСЂРµР¶РµРЅРЅСЏ HTML Р·РІС–С‚Сѓ

        РџРѕРІРµСЂС‚Р°С”:
        -----------
        СЂРµР·СѓР»СЊС‚Р°С‚Рё РїСЂРѕС„С–Р»СЋРІР°РЅРЅСЏ
        '''
        try:
            from pyinstrument import Profiler

            # РџРµСЂРµРІРµРґРµРЅРЅСЏ РјРѕРґРµР»С– РЅР° CPU (pyinstrument РЅРµ РїС–РґС‚СЂРёРјСѓС” CUDA)
            device = torch.device("cpu")
            model = self.model.to(device)
            model.eval()

            # РћС‚СЂРёРјР°РЅРЅСЏ Р±Р°С‚С‡Сѓ РґР»СЏ С–РЅС„РµСЂРµРЅСЃСѓ
            for inputs, _ in self.test_loader:
                inputs = inputs.to(device)
                break

            # РќР°Р»Р°С€С‚СѓРІР°РЅРЅСЏ С€Р»СЏС…Сѓ РґР»СЏ HTML
            if html_path is None:
                html_path = os.path.join(self.save_dir, f"pyinstrument_profile_{int(time.time())}.html")

            # Р РѕР·С–РіСЂС–РІ
            with torch.no_grad():
                for _ in range(5):
                    _ = model(inputs)

            # РџСЂРѕС„С–Р»СЋРІР°РЅРЅСЏ
            logger.info("РџСЂРѕС„С–Р»СЋРІР°РЅРЅСЏ РјРѕРґРµР»С– Р· pyinstrument...")
            profiler = Profiler()
            profiler.start()

            with torch.no_grad():
                _ = model(inputs)

            profiler.stop()

            # Р—Р±РµСЂРµР¶РµРЅРЅСЏ HTML Р·РІС–С‚Сѓ
            with open(html_path, 'w') as f:
                f.write(profiler.output_html())

            logger.info(f"HTML Р·РІС–С‚ РїСЂРѕС„С–Р»СЋРІР°РЅРЅСЏ Р·Р±РµСЂРµР¶РµРЅРѕ Сѓ {html_path}")

            return profiler

        except ImportError:
            logger.error("РќРµ РІРґР°Р»РѕСЃСЏ С–РјРїРѕСЂС‚СѓРІР°С‚Рё pyinstrument. Р’СЃС‚Р°РЅРѕРІС–С‚СЊ Р№РѕРіРѕ Р·Р° РґРѕРїРѕРјРѕРіРѕСЋ 'pip install pyinstrument'")
            return None
        except Exception as e:
            logger.error(f"РџРѕРјРёР»РєР° РїСЂРѕС„С–Р»СЋРІР°РЅРЅСЏ Р· pyinstrument: {e}")
            return None

    def profile_with_cprofile(self, output_path=None):
        '''
        РџСЂРѕС„С–Р»СЋРІР°РЅРЅСЏ РјРѕРґРµР»С– Р·Р° РґРѕРїРѕРјРѕРіРѕСЋ cProfile

        РџР°СЂР°РјРµС‚СЂРё:
        -----------
        output_path: С€Р»СЏС… РґР»СЏ Р·Р±РµСЂРµР¶РµРЅРЅСЏ СЂРµР·СѓР»СЊС‚Р°С‚С–РІ РїСЂРѕС„С–Р»СЋРІР°РЅРЅСЏ

        РџРѕРІРµСЂС‚Р°С”:
        -----------
        СЂРµР·СѓР»СЊС‚Р°С‚Рё РїСЂРѕС„С–Р»СЋРІР°РЅРЅСЏ
        '''
        try:
            import cProfile
            import pstats
            import io

            # РџРµСЂРµРІРµРґРµРЅРЅСЏ РјРѕРґРµР»С– РЅР° CPU
            device = torch.device("cpu")
            model = self.model.to(device)
            model.eval()

            # РћС‚СЂРёРјР°РЅРЅСЏ Р±Р°С‚С‡Сѓ РґР»СЏ С–РЅС„РµСЂРµРЅСЃСѓ
            for inputs, _ in self.test_loader:
                inputs = inputs.to(device)
                break

            # РќР°Р»Р°С€С‚СѓРІР°РЅРЅСЏ С€Р»СЏС…Сѓ РґР»СЏ СЂРµР·СѓР»СЊС‚Р°С‚С–РІ
            if output_path is None:
                output_path = os.path.join(self.save_dir, f"cprofile_stats_{int(time.time())}.txt")

            # Р РѕР·С–РіСЂС–РІ
            with torch.no_grad():
                for _ in range(5):
                    _ = model(inputs)

            # РџСЂРѕС„С–Р»СЋРІР°РЅРЅСЏ
            logger.info("РџСЂРѕС„С–Р»СЋРІР°РЅРЅСЏ РјРѕРґРµР»С– Р· cProfile...")
            profiler = cProfile.Profile()
            profiler.enable()

            with torch.no_grad():
                _ = model(inputs)

            profiler.disable()

            # Р—Р±РµСЂРµР¶РµРЅРЅСЏ СЂРµР·СѓР»СЊС‚Р°С‚С–РІ
            s = io.StringIO()
            ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
            ps.print_stats(50)  # РўРѕРї-50 С„СѓРЅРєС†С–Р№ Р·Р° РєСѓРјСѓР»СЏС‚РёРІРЅРёРј С‡Р°СЃРѕРј

            with open(output_path, 'w') as f:
                f.write(s.getvalue())

            logger.info(f"cProfile Р·РІС–С‚ Р·Р±РµСЂРµР¶РµРЅРѕ Сѓ {output_path}")

            return profiler

        except Exception as e:
            logger.error(f"РџРѕРјРёР»РєР° РїСЂРѕС„С–Р»СЋРІР°РЅРЅСЏ Р· cProfile: {e}")
            return None

    def run_all_profilers(self, model_name="model"):
        '''
        Р—Р°РїСѓСЃРє РІСЃС–С… РґРѕСЃС‚СѓРїРЅРёС… РїСЂРѕС„С–Р»СЋРІР°Р»СЊРЅРёРєС–РІ

        РџР°СЂР°РјРµС‚СЂРё:
        -----------
        model_name: РЅР°Р·РІР° РјРѕРґРµР»С– РґР»СЏ С–РјРµРЅСѓРІР°РЅРЅСЏ С„Р°Р№Р»С–РІ

        РџРѕРІРµСЂС‚Р°С”:
        -----------
        СЃР»РѕРІРЅРёРє Р· СЂРµР·СѓР»СЊС‚Р°С‚Р°РјРё РїСЂРѕС„С–Р»СЋРІР°РЅРЅСЏ
        '''
        results = {}

        # РџСЂРѕС„С–Р»СЋРІР°РЅРЅСЏ Р· torch.profiler
        trace_path = os.path.join(self.save_dir, f"{model_name}_torch_trace_{int(time.time())}")
        results["torch_profiler"] = self.profile_with_torch_profiler(trace_path=trace_path)

        # РџСЂРѕС„С–Р»СЋРІР°РЅРЅСЏ Р· pyinstrument
        html_path = os.path.join(self.save_dir, f"{model_name}_pyinstrument_{int(time.time())}.html")
        results["pyinstrument"] = self.profile_with_pyinstrument(html_path=html_path)

        # РџСЂРѕС„С–Р»СЋРІР°РЅРЅСЏ Р· cProfile
        cprofile_path = os.path.join(self.save_dir, f"{model_name}_cprofile_{int(time.time())}.txt")
        results["cprofile"] = self.profile_with_cprofile(output_path=cprofile_path)

        return results

def main():
    parser = argparse.ArgumentParser(description="РЎС‚РІРѕСЂРµРЅРЅСЏ flamegraph РґР»СЏ РјРѕРґРµР»РµР№ РјР°С€РёРЅРЅРѕРіРѕ РЅР°РІС‡Р°РЅРЅСЏ")
    parser.add_argument("--model", type=str, required=True,
                        help="РЁР»СЏС… РґРѕ РјРѕРґРµР»С– РґР»СЏ РїСЂРѕС„С–Р»СЋРІР°РЅРЅСЏ")
    parser.add_argument("--dataset", type=str, default=None,
                        help="РЁР»СЏС… РґРѕ С‚РµСЃС‚РѕРІРѕРіРѕ РґР°С‚Р°СЃРµС‚Сѓ")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Р РѕР·РјС–СЂ Р±Р°С‚С‡Сѓ РґР»СЏ С–РЅС„РµСЂРµРЅСЃСѓ")
    parser.add_argument("--save-dir", type=str, default="profile_results",
                        help="Р”РёСЂРµРєС‚РѕСЂС–СЏ РґР»СЏ Р·Р±РµСЂРµР¶РµРЅРЅСЏ СЂРµР·СѓР»СЊС‚Р°С‚С–РІ")
    parser.add_argument("--use-cuda", action="store_true",
                        help="Р’РёРєРѕСЂРёСЃС‚РѕРІСѓРІР°С‚Рё CUDA, СЏРєС‰Рѕ РґРѕСЃС‚СѓРїРЅРѕ")
    parser.add_argument("--profiler", type=str, choices=["all", "torch", "pyinstrument", "cprofile"], default="all",
                        help="РЇРєРёР№ РїСЂРѕС„С–Р»СЋРІР°Р»СЊРЅРёРє РІРёРєРѕСЂРёСЃС‚РѕРІСѓРІР°С‚Рё")

    args = parser.parse_args()

    # Р—Р°РІР°РЅС‚Р°Р¶РµРЅРЅСЏ РјРѕРґРµР»С–
    try:
        logger.info(f"Р—Р°РІР°РЅС‚Р°Р¶РµРЅРЅСЏ РјРѕРґРµР»С– Р· {args.model}")
        model = torch.load(args.model, map_location=torch.device('cpu'))
        model.eval()
    except Exception as e:
        logger.error(f"РџРѕРјРёР»РєР° Р·Р°РІР°РЅС‚Р°Р¶РµРЅРЅСЏ РјРѕРґРµР»С–: {e}")
        return

    # РЎС‚РІРѕСЂРµРЅРЅСЏ РїСЂРѕС„С–Р»СЋРІР°Р»СЊРЅРёРєР°
    profiler = FlameGraphProfiler(
        model=model,
        dataset_path=args.dataset,
        batch_size=args.batch_size,
        save_dir=args.save_dir
    )

    # Р—Р°РїСѓСЃРє РїСЂРѕС„С–Р»СЋРІР°РЅРЅСЏ
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

    logger.info("РџСЂРѕС„С–Р»СЋРІР°РЅРЅСЏ Р·Р°РІРµСЂС€РµРЅРѕ")

if __name__ == "__main__":
    main()

