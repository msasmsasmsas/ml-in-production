# Скрипт для пошуку гіперпараметрів за допомогою Weights & Biases

import wandb
from train import train_model

def hyperparameter_search():
    # Конфігурація для пошуку гіперпараметрів
    sweep_config = {
        "method": "grid",
        "metric": {"name": "eval_loss", "goal": "minimize"},
        "parameters": {
            "learning_rate": {"values": [1e-5, 2e-5, 3e-5]},
            "batch_size": {"values": [8, 16]},
            "epochs": {"values": [3, 4]},
            "weight_decay": {"values": [0.01, 0.1]},
        },
    }

    # Ініціалізація sweep
    sweep_id = wandb.sweep(sweep_config, project="ml-in-production-hw5")

    def train():
        # Запуск тренування з конфігурацією sweep
        with wandb.init(project="ml-in-production-hw5", name="hyperparam-search"):
            config = wandb.config
            metrics = train_model(config)
            wandb.log(metrics)

    # Запуск агента для виконання пошуку
    wandb.agent(sweep_id, function=train)

if __name__ == "__main__":
    hyperparameter_search()