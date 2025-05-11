# Конвертація датасету в StreamingDataset
from datasets import Dataset, IterableDataset
import pandas as pd
import numpy as np

def generate_sample_data(size=1000):
    # Генерація тестового датасету
    data = {
        "id": range(size),
        "value": np.random.randn(size),
        "category": np.random.choice(["A", "B", "C"], size)
    }
    return pd.DataFrame(data)

def create_streaming_dataset():
    # Створення StreamingDataset
    df = generate_sample_data()
    dataset = Dataset.from_pandas(df)
    streaming_dataset = IterableDataset.from_generator(
        lambda: ({"id": row["id"], "value": row["value"], "category": row["category"]} for _, row in df.iterrows())
    )
    return streaming_dataset

if __name__ == "__main__":
    # Тестування потокового датасету
    streaming_dataset = create_streaming_dataset()
    for example in streaming_dataset:
        print(example)
        break  # Вивести перший приклад