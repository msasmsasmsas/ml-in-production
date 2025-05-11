# Модуль для завантаження та попередньої обробки даних

import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer


def load_dataset(data_path):
    # Завантаження датасету з CSV-файлу
    df = pd.read_csv(data_path)
    dataset = Dataset.from_pandas(df)
    return dataset


def preprocess_data(dataset, tokenizer_name="distilbert-base-uncased"):
    # Попередня обробка даних: токенізація текстів
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def tokenize_function(examples):
        # Токенізація з вирівнюванням довжини та обрізанням
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    return tokenized_dataset