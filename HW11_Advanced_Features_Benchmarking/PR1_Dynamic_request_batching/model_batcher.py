import numpy as np
import torch
import time
import queue
import threading
from typing import List, Dict, Any, Optional

class DynamicBatcher:
    """
    Клас для динамічного пакетування запитів для моделі
    """
    def __init__(self, model, max_batch_size: int = 16, max_wait_time: float = 0.1):
        """
        Ініціалізація батчера

        Параметри:
        -----------
        model: об'єкт моделі, який може приймати батчі
        max_batch_size: максимальний розмір батчу
        max_wait_time: максимальний час очікування в секундах
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
        Фоновий процес, який збирає запити і обробляє їх пакетами
        """
        while self.running:
            batch = []
            start_time = time.time()
            request_item = None

            # Очікуємо перший запит
            try:
                request_item = self.request_queue.get(timeout=1.0)
                batch.append(request_item)
            except queue.Empty:
                continue

            # Збираємо додаткові запити, поки не досягнемо обмежень
            try:
                while len(batch) < self.max_batch_size and time.time() - start_time < self.max_wait_time:
                    timeout = max(0, self.max_wait_time - (time.time() - start_time))
                    request_item = self.request_queue.get(timeout=timeout)
                    batch.append(request_item)
            except queue.Empty:
                pass  # Час очікування минув, обробляємо зібраний батч

            if batch:  # Якщо є запити для обробки
                self._process_batch(batch)

    def _process_batch(self, batch: List[Dict[str, Any]]):
        """
        Обробка батчу запитів

        Параметри:
        -----------
        batch: список словників з запитами та колбеками
        """
        # Підготовка вхідних даних
        inputs = [item['data'] for item in batch]

        # Перетворення списку входів в один тензор
        try:
            if isinstance(inputs[0], np.ndarray):
                batched_input = np.stack(inputs)
            elif isinstance(inputs[0], torch.Tensor):
                batched_input = torch.stack(inputs)
            else:
                # Інші типи даних
                batched_input = inputs

            # Виконання прогнозу моделі
            with torch.no_grad():
                results = self.model(batched_input)

            # Розподіл результатів по запитах
            for i, item in enumerate(batch):
                if isinstance(results, torch.Tensor):
                    result = results[i].cpu().numpy()
                elif isinstance(results, np.ndarray):
                    result = results[i]
                else:
                    result = results[i]

                # Виклик колбека з результатом
                item['callback'](result, None)  # результат, помилка=None

        except Exception as e:
            # Обробка помилок
            for item in batch:
                item['callback'](None, str(e))  # результат=None, помилка

    def predict_async(self, data, callback):
        """
        Асинхронне додавання запиту на прогнозування

        Параметри:
        -----------
        data: вхідні дані для моделі
        callback: функція, яка буде викликана з результатом
        """
        self.request_queue.put({
            'data': data,
            'callback': callback
        })

    def predict(self, data):
        """
        Синхронне прогнозування (обгортка над асинхронним)

        Параметри:
        -----------
        data: вхідні дані для моделі

        Повертає:
        -----------
        результат прогнозування
        """
        result_container = {'result': None, 'error': None, 'done': False}

        def _callback(result, error):
            result_container['result'] = result
            result_container['error'] = error
            result_container['done'] = True

        self.predict_async(data, _callback)

        # Очікування завершення
        while not result_container['done']:
            time.sleep(0.001)

        if result_container['error']:
            raise RuntimeError(result_container['error'])

        return result_container['result']

    def shutdown(self):
        """
        Зупинка обробки та очищення ресурсів
        """
        self.running = False
        if self.worker_thread.is_alive():
            self.worker_thread.join(timeout=5.0)
