# Updated version for PR
import numpy as np
import torch
import time
import queue
import threading
from typing import List, Dict, Any, Optional

class DynamicBatcher:
    """
    РљР»Р°СЃ РґР»СЏ РґРёРЅР°РјС–С‡РЅРѕРіРѕ РїР°РєРµС‚СѓРІР°РЅРЅСЏ Р·Р°РїРёС‚С–РІ РґР»СЏ РјРѕРґРµР»С–
    """
    def __init__(self, model, max_batch_size: int = 16, max_wait_time: float = 0.1):
        """
        Р†РЅС–С†С–Р°Р»С–Р·Р°С†С–СЏ Р±Р°С‚С‡РµСЂР°

        РџР°СЂР°РјРµС‚СЂРё:
        -----------
        model: РѕР±'С”РєС‚ РјРѕРґРµР»С–, СЏРєРёР№ РјРѕР¶Рµ РїСЂРёР№РјР°С‚Рё Р±Р°С‚С‡С–
        max_batch_size: РјР°РєСЃРёРјР°Р»СЊРЅРёР№ СЂРѕР·РјС–СЂ Р±Р°С‚С‡Сѓ
        max_wait_time: РјР°РєСЃРёРјР°Р»СЊРЅРёР№ С‡Р°СЃ РѕС‡С–РєСѓРІР°РЅРЅСЏ РІ СЃРµРєСѓРЅРґР°С…
        """
        self.model = model
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        self.request_queue = queue.Queue()
        self.running = True
        self.worker_thread = threading.Thread(target=self._batch_processor)
        self.worker_thread.daemon = True
        self.worker_thread.start()

    def _batch_processor(self):
        """
        Р¤РѕРЅРѕРІРёР№ РїСЂРѕС†РµСЃ, СЏРєРёР№ Р·Р±РёСЂР°С” Р·Р°РїРёС‚Рё С– РѕР±СЂРѕР±Р»СЏС” С—С… РїР°РєРµС‚Р°РјРё
        """
        while self.running:
            batch = []
            start_time = time.time()
            request_item = None

            # РћС‡С–РєСѓС”РјРѕ РїРµСЂС€РёР№ Р·Р°РїРёС‚
            try:
                request_item = self.request_queue.get(timeout=1.0)
                batch.append(request_item)
            except queue.Empty:
                continue

            # Р—Р±РёСЂР°С”РјРѕ РґРѕРґР°С‚РєРѕРІС– Р·Р°РїРёС‚Рё, РїРѕРєРё РЅРµ РґРѕСЃСЏРіРЅРµРјРѕ РѕР±РјРµР¶РµРЅСЊ
            try:
                while len(batch) < self.max_batch_size and time.time() - start_time < self.max_wait_time:
                    timeout = max(0, self.max_wait_time - (time.time() - start_time))
                    request_item = self.request_queue.get(timeout=timeout)
                    batch.append(request_item)
            except queue.Empty:
                pass  # Р§Р°СЃ РѕС‡С–РєСѓРІР°РЅРЅСЏ РјРёРЅСѓРІ, РѕР±СЂРѕР±Р»СЏС”РјРѕ Р·С–Р±СЂР°РЅРёР№ Р±Р°С‚С‡

            if batch:  # РЇРєС‰Рѕ С” Р·Р°РїРёС‚Рё РґР»СЏ РѕР±СЂРѕР±РєРё
                self._process_batch(batch)

    def _process_batch(self, batch: List[Dict[str, Any]]):
        """
        РћР±СЂРѕР±РєР° Р±Р°С‚С‡Сѓ Р·Р°РїРёС‚С–РІ

        РџР°СЂР°РјРµС‚СЂРё:
        -----------
        batch: СЃРїРёСЃРѕРє СЃР»РѕРІРЅРёРєС–РІ Р· Р·Р°РїРёС‚Р°РјРё С‚Р° РєРѕР»Р±РµРєР°РјРё
        """
        # РџС–РґРіРѕС‚РѕРІРєР° РІС…С–РґРЅРёС… РґР°РЅРёС…
        inputs = [item['data'] for item in batch]

        # РџРµСЂРµС‚РІРѕСЂРµРЅРЅСЏ СЃРїРёСЃРєСѓ РІС…РѕРґС–РІ РІ РѕРґРёРЅ С‚РµРЅР·РѕСЂ
        try:
            if isinstance(inputs[0], np.ndarray):
                batched_input = np.stack(inputs)
            elif isinstance(inputs[0], torch.Tensor):
                batched_input = torch.stack(inputs)
            else:
                # Р†РЅС€С– С‚РёРїРё РґР°РЅРёС…
                batched_input = inputs

            # Р’РёРєРѕРЅР°РЅРЅСЏ РїСЂРѕРіРЅРѕР·Сѓ РјРѕРґРµР»С–
            with torch.no_grad():
                results = self.model(batched_input)

            # Р РѕР·РїРѕРґС–Р» СЂРµР·СѓР»СЊС‚Р°С‚С–РІ РїРѕ Р·Р°РїРёС‚Р°С…
            for i, item in enumerate(batch):
                if isinstance(results, torch.Tensor):
                    result = results[i].cpu().numpy()
                elif isinstance(results, np.ndarray):
                    result = results[i]
                else:
                    result = results[i]

                # Р’РёРєР»РёРє РєРѕР»Р±РµРєР° Р· СЂРµР·СѓР»СЊС‚Р°С‚РѕРј
                item['callback'](result, None)  # СЂРµР·СѓР»СЊС‚Р°С‚, РїРѕРјРёР»РєР°=None

        except Exception as e:
            # РћР±СЂРѕР±РєР° РїРѕРјРёР»РѕРє
            for item in batch:
                item['callback'](None, str(e))  # СЂРµР·СѓР»СЊС‚Р°С‚=None, РїРѕРјРёР»РєР°

    def predict_async(self, data, callback):
        """
        РђСЃРёРЅС…СЂРѕРЅРЅРµ РґРѕРґР°РІР°РЅРЅСЏ Р·Р°РїРёС‚Сѓ РЅР° РїСЂРѕРіРЅРѕР·СѓРІР°РЅРЅСЏ

        РџР°СЂР°РјРµС‚СЂРё:
        -----------
        data: РІС…С–РґРЅС– РґР°РЅС– РґР»СЏ РјРѕРґРµР»С–
        callback: С„СѓРЅРєС†С–СЏ, СЏРєР° Р±СѓРґРµ РІРёРєР»РёРєР°РЅР° Р· СЂРµР·СѓР»СЊС‚Р°С‚РѕРј
        """
        self.request_queue.put({
            'data': data,
            'callback': callback
        })

    def predict(self, data):
        """
        РЎРёРЅС…СЂРѕРЅРЅРµ РїСЂРѕРіРЅРѕР·СѓРІР°РЅРЅСЏ (РѕР±РіРѕСЂС‚РєР° РЅР°Рґ Р°СЃРёРЅС…СЂРѕРЅРЅРёРј)

        РџР°СЂР°РјРµС‚СЂРё:
        -----------
        data: РІС…С–РґРЅС– РґР°РЅС– РґР»СЏ РјРѕРґРµР»С–

        РџРѕРІРµСЂС‚Р°С”:
        -----------
        СЂРµР·СѓР»СЊС‚Р°С‚ РїСЂРѕРіРЅРѕР·СѓРІР°РЅРЅСЏ
        """
        result_container = {'result': None, 'error': None, 'done': False}

        def _callback(result, error):
            result_container['result'] = result
            result_container['error'] = error
            result_container['done'] = True

        self.predict_async(data, _callback)

        # РћС‡С–РєСѓРІР°РЅРЅСЏ Р·Р°РІРµСЂС€РµРЅРЅСЏ
        while not result_container['done']:
            time.sleep(0.001)

        if result_container['error']:
            raise RuntimeError(result_container['error'])

        return result_container['result']

    def shutdown(self):
        """
        Р—СѓРїРёРЅРєР° РѕР±СЂРѕР±РєРё С‚Р° РѕС‡РёС‰РµРЅРЅСЏ СЂРµСЃСѓСЂСЃС–РІ
        """
        self.running = False
        if self.worker_thread.is_alive():
            self.worker_thread.join(timeout=5.0)

