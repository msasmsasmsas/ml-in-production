# Бенчмаркінг форматів Pandas для збереження та завантаження даних
import pandas as pd
import numpy as np
import time
import h5py
import pyarrow
from pathlib import Path


def generate_dataset(size=100000):
    # Генерація синтетичного набору даних
    data = {
        "id": range(size),
        "value": np.random.randn(size),
        "category": np.random.choice(["A", "B", "C"], size)
    }
    return pd.DataFrame(data)


def benchmark_formats():
    # Налаштування
    dataset = generate_dataset()
    formats = ["csv", "parquet", "hdf5"]
    results = {"format": [], "save_time": [], "load_time": []}

    # Тестування кожного формату
    for fmt in formats:
        file_path = Path(f"test_data.{fmt}")

        # Збереження
        start_time = time.time()
        if fmt == "csv":
            dataset.to_csv(file_path, index=False)
        elif fmt == "parquet":
            dataset.to_parquet(file_path, index=False)
        elif fmt == "hdf5":
            dataset.to_hdf(file_path, key="data", mode="w")
        save_time = time.time() - start_time

        # Завантаження
        start_time = time.time()
        if fmt == "csv":
            pd.read_csv(file_path)
        elif fmt == "parquet":
            pd.read_parquet(file_path)
        elif fmt == "hdf5":
            pd.read_hdf(file_path, key="data")
        load_time = time.time() - start_time

        # Збір результатів
        results["format"].append(fmt)
        results["save_time"].append(save_time)
        results["load_time"].append(load_time)

        # Видалення файлу
        file_path.unlink()

    return pd.DataFrame(results)


if __name__ == "__main__":
    # Виконання бенчмаркінгу
    results = benchmark_formats()
    print(results)