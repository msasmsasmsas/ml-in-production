# Скрипт для попереднього тренування моделі MosaicBERT

from transformers import BertConfig, BertForPreTraining, Trainer, TrainingArguments
from datasets import load_dataset
import wandb

def train_mosaicbert():
    # Ініціалізація Weights & Biases
    wandb.init(project="ml-in-production-hw5", name="mosaicbert")

    # Завантаження датасету (наприклад, Wikipedia)
    dataset = load_dataset("wikipedia", "20220301.en", split="train[:10000]")

    # Конфігурація моделі MosaicBERT
    config = BertConfig(
        vocab_size=30522,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
    )
    model = BertForPreTraining(config=config)

    # Налаштування параметрів тренування
    training_args = TrainingArguments(
        output_dir="./mosaicbert_results",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        num_train_epochs=3,
        logging_dir="./logs",
        logging_steps=10,
        report_to="wandb",
    )

    # Ініціалізація тренера
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    # Тренування моделі
    trainer.train()
    wandb.finish()

if __name__ == "__main__":
    train_mosaicbert()