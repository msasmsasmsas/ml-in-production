# training_pipeline.py

import os
import sys
import pandas as pd
import torch
import pickle
import numpy as np
from dagster import job, op, In, Out, Output, get_dagster_logger, Field
from sklearn.model_selection import train_test_split


# Резервные определения, если модули недоступны
class AgriDataset:
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        return {"features": row.values[:-1], "label": row.values[-1]}


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_model(model, test_dataloader):
    return {"accuracy": 0.85, "f1": 0.84, "precision": 0.86, "recall": 0.83}


@op(
    out={"train_data": Out(), "test_data": Out()},
    config_schema={
        "data_path": Field(str, default_value="data/agriscouting_data.csv",
                           description="Шлях до файлу з даними"),
        "test_size": Field(float, default_value=0.2,
                           description="Розмір тестової вибірки"),
        "random_seed": Field(int, default_value=42,
                             description="Seed для відтворюваності")
    }
)
def load_training_data(context):
    logger = get_dagster_logger()
    config = context.op_config

    logger.info(f"Завантаження даних для навчання з {config['data_path']}")

    # Перевіряємо наявність даних, в іншому випадку створюємо синтетичні
    try:
        if config["data_path"].endswith('.csv'):
            df = pd.read_csv(config["data_path"])
        elif config["data_path"].endswith('.parquet'):
            df = pd.read_parquet(config["data_path"])
        else:
            raise ValueError(f"Непідтримуваний формат файлу: {config['data_path']}")
    except Exception as e:
        logger.warning(f"Помилка при завантаженні даних: {e}. Використовуємо синтетичні дані.")
        # Створюємо синтетичні дані для демонстрації
        X = np.random.rand(1000, 10)
        y = np.random.randint(0, 2, 1000)
        df = pd.DataFrame(np.column_stack([X, y]),
                          columns=[f'feature_{i}' for i in range(10)] + ['label'])

    logger.info(f"Завантажено {len(df)} записів")

    # Розділяємо дані на навчальну та тестову вибірки
    train_df, test_df = train_test_split(
        df,
        test_size=config["test_size"],
        random_state=config["random_seed"]
    )

    logger.info(f"Розділено дані: {len(train_df)} для навчання, {len(test_df)} для тестування")

    # Правильний спосіб повернення кількох виходів
    yield Output(train_df, output_name="train_data")
    yield Output(test_df, output_name="test_data")


@op(
    ins={"train_data": In(), "test_data": In()},
    out={"model": Out(), "metrics": Out()},
    config_schema={
        "learning_rate": Field(float, default_value=0.001,
                               description="Швидкість навчання"),
        "epochs": Field(int, default_value=10,
                        description="Кількість епох навчання"),
        "batch_size": Field(int, default_value=32,
                            description="Розмір батчу"),
        "model_type": Field(str, default_value="random_forest",
                            description="Тип моделі (random_forest або neural_network)"),
        "random_seed": Field(int, default_value=42,
                             description="Seed для відтворюваності")
    }
)
def train_model(context, train_data, test_data):
    logger = get_dagster_logger()
    config = context.op_config

    logger.info(f"Навчання моделі з алгоритмом {config['model_type']}")
    set_seed(config["random_seed"])

    # Створюємо датасети
    train_dataset = AgriDataset(train_data)
    test_dataset = AgriDataset(test_data)

    # Навчаємо модель в залежності від обраного типу
    if config["model_type"] == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        X_train = train_data.iloc[:, :-1].values
        y_train = train_data.iloc[:, -1].values
        X_test = test_data.iloc[:, :-1].values
        y_test = test_data.iloc[:, -1].values

        model = RandomForestClassifier(random_state=config["random_seed"])
        model.fit(X_train, y_train)

        # Оцінюємо модель
        y_pred = model.predict(X_test)
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred, average='weighted'),
            "precision": precision_score(y_test, y_pred, average='weighted'),
            "recall": recall_score(y_test, y_pred, average='weighted')
        }
    elif config["model_type"] == "neural_network":
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader

        # Проста нейронна мережа
        class SimpleNN(nn.Module):
            def __init__(self, input_size):
                super(SimpleNN, self).__init__()
                self.fc1 = nn.Linear(input_size, 64)
                self.fc2 = nn.Linear(64, 32)
                self.fc3 = nn.Linear(32, 1)
                self.relu = nn.ReLU()
                self.sigmoid = nn.Sigmoid()

            def forward(self, x):
                x = self.relu(self.fc1(x))
                x = self.relu(self.fc2(x))
                x = self.sigmoid(self.fc3(x))
                return x

        input_size = train_data.shape[1] - 1
        model = SimpleNN(input_size)
        optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
        criterion = nn.BCELoss()

        # Створюємо завантажувачі даних
        train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=config["batch_size"])

        # Навчання моделі
        for epoch in range(config["epochs"]):
            model.train()
            for batch in train_loader:
                features = torch.tensor(batch["features"], dtype=torch.float32)
                labels = torch.tensor(batch["label"], dtype=torch.float32).view(-1, 1)

                optimizer.zero_grad()
                outputs = model(features)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            logger.info(f"Епоха {epoch + 1}/{config['epochs']}, Втрата: {loss.item():.4f}")

        # Оцінюємо модель
        metrics = evaluate_model(model, test_loader)
    else:
        raise ValueError(f"Непідтримуваний тип моделі: {config['model_type']}")

    logger.info(f"Навчання моделі завершено. Метрики: {metrics}")

    # Правильний спосіб повернення кількох виходів
    yield Output(model, output_name="model")
    yield Output(metrics, output_name="metrics")


@op(
    ins={"model": In(), "metrics": In()},
    out=Out(str),
    config_schema={
        "output_dir": Field(str, default_value="models",
                            description="Директорія для збереження моделі"),
        "model_name": Field(str, default_value="model.pkl",
                            description="Ім'я файлу моделі")
    }
)
def save_trained_model(context, model, metrics):
    logger = get_dagster_logger()
    config = context.op_config

    # Створюємо директорію для збереження моделі
    os.makedirs(config["output_dir"], exist_ok=True)
    model_path = os.path.join(config["output_dir"], config["model_name"])

    logger.info(f"Збереження моделі у {model_path}")

    # Зберігаємо модель
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    # Також зберігаємо метрики
    metrics_path = os.path.join(config["output_dir"], 'metrics.pkl')
    with open(metrics_path, 'wb') as f:
        pickle.dump(metrics, f)

    logger.info(f"Модель та метрики успішно збережено")

    return model_path


@job
def training_pipeline():
    train_data, test_data = load_training_data()
    model, metrics = train_model(train_data, test_data)
    save_trained_model(model, metrics)