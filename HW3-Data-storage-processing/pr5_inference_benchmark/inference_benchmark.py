# Бенчмаркінг інференсу: один vs кілька процесів
import torch
import torch.nn as nn
import time
import multiprocessing as mp
import numpy as np


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(100, 10)

    def forward(self, x):
        return self.fc(x)


def single_process_inference(model, data, iterations):
    # Інференс в одному процесі
    start_time = time.time()
    for _ in range(iterations):
        with torch.no_grad():
            model(data)
    return time.time() - start_time


def worker(model, data, iters, queue):
    # Робоча функція для багатопроцесного інференсу
    time_taken = single_process_inference(model, data, iters)
    queue.put(time_taken)


def multi_process_inference(model, data, iterations, num_processes):
    # Інференс у кількох процесах
    start_time = time.time()
    queue = mp.Queue()
    processes = []
    iters_per_process = iterations // num_processes

    for _ in range(num_processes):
        p = mp.Process(target=worker, args=(model, data, iters_per_process, queue))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    total_time = sum(queue.get() for _ in range(num_processes))
    return time.time() - start_time


if __name__ == "__main__":
    # Налаштування
    model = SimpleModel()
    data = torch.randn(1000, 100)
    iterations = 1000
    num_processes = 4

    # Виконання бенчмаркінгу
    single_time = single_process_inference(model, data, iterations)
    multi_time = multi_process_inference(model, data, iterations, num_processes)

    print(f"Час одного процесу: {single_time:.2f} сек")
    print(f"Час кількох процесів ({num_processes}): {multi_time:.2f} сек")