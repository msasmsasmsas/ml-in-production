# Скрипт для розподіленого тренування моделі з використанням PyTorch, Accelerate та Ray

from accelerate import Accelerator
from transformers import Trainer, TrainingArguments
from dataset import load_dataset, preprocess_data
from model import get_model
import ray
from ray import train as ray_train

def distributed_train():
    # Ініціалізація Ray
    ray.init()

    # Ініціалізація Accelerate
    accelerator = Accelerator()

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
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
    )

    # Підготовка моделі та даних для розподіленого тренування
    model, train_dataset, eval_dataset = accelerator.prepare(model, train_dataset, eval_dataset)

    # Ініціалізація тренера
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    # Тренування моделі
    trainer.train()

    # Збереження моделі
    trainer.save_model("./trained_model")

if __name__ == "__main__":
    distributed_train()