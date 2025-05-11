# Скрипт для тренування моделі з підтримкою пошуку гіперпараметрів

import torch
from transformers import Trainer, TrainingArguments
from dataset import load_dataset, preprocess_data
from model import get_model

def train_model(config):
    # Завантаження та попередня обробка даних
    data_path = "../data/dataset.csv"
    dataset = load_dataset(data_path)
    tokenized_dataset = preprocess_data(dataset)

    # Розділення на тренувальну та тестову вибірки
    train_test_split = tokenized_dataset.train_test_split(test_size=0.2)
    train_dataset = train_test_split["train"]
    eval_dataset = train_test_split["test"]

    # Завантаження моделі
    model = get_model(num_labels=3)

    # Налаштування параметрів тренування
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=config["learning_rate"],
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"],
        num_train_epochs=config["epochs"],
        weight_decay=config["weight_decay"],
        logging_dir="./logs",
        logging_steps=10,
        report_to="wandb",
    )

    # Ініціалізація тренера
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    # Тренування моделі
    trainer.train()
    return trainer.evaluate()