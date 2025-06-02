import os
import argparse
import torch
import wandb
import logging
from datetime import datetime
from torchvision import models
import torch.nn as nn

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("wandb_model_store")


def init_wandb(project_name="diseases-classification"):
    """
    Инициализация Weights & Biases
    """
    try:
        # Проверяем, есть ли файл .env с ключом API
        if os.path.exists(".env"):
            with open(".env", "r") as f:
                for line in f:
                    if line.startswith("wandb"):
                        api_key = line.split("=")[1].strip()
                        os.environ["WANDB_API_KEY"] = api_key
                        break

        # Инициализация wandb
        wandb.init(project=project_name)
        logger.info(f"W&B инициализирован для проекта: {project_name}")
        return True
    except Exception as e:
        logger.error(f"Ошибка при инициализации W&B: {e}")
        return False


def load_model(model_path, num_classes=102):
    """
    Загрузка модели из файла
    """
    try:
        # Загрузка state_dict
        state_dict = torch.load(model_path)

        # Если загружен не state_dict, а сама модель
        if hasattr(state_dict, 'eval'):
            model = state_dict
            logger.info(f"Загружена готовая модель из {model_path}")
            return model

        # Создание новой модели с правильным числом классов
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

        # Загрузка весов
        model.load_state_dict(state_dict)
        model.eval()

        logger.info(f"Модель ResNet18 загружена из {model_path}")
        return model
    except Exception as e:
        logger.error(f"Ошибка при загрузке модели: {e}")
        return None


def store_model_in_wandb(model, model_path, metadata=None):
    """
    Сохранение модели в Weights & Biases Model Registry

    Args:
        model: PyTorch модель
        model_path: Путь к исходному файлу модели
        metadata: Метаданные модели
    """
    try:
        # Получаем имя файла из пути
        model_name = os.path.basename(model_path)

        # Создаем временный файл для сохранения модели в формате torchscript
        # Используем tempfile для кроссплатформенного создания временного файла
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as temp_file:
            tmp_model_path = temp_file.name

        # Экспортируем модель в формат TorchScript
        scripted_model = torch.jit.script(model)
        torch.jit.save(scripted_model, tmp_model_path)

        # Если метаданные не переданы, создаем базовые
        if metadata is None:
            metadata = {
                "framework": "pytorch",
                "date": datetime.now().strftime("%Y-%m-%d"),
                "model_type": "ResNet18",
                "dataset": "Agricultural Diseases"
            }

        # Логируем артефакт модели
        artifact = wandb.Artifact(
            name=f"model-{wandb.run.id}",
            type="model",
            metadata=metadata
        )

        # Добавляем файл модели в артефакт
        artifact.add_file(tmp_model_path, name="model.pt")

        # Загружаем артефакт в W&B
        wandb.log_artifact(artifact)

        logger.info(f"Модель успешно сохранена в W&B Model Registry как '{artifact.name}'")

        # Удаляем временный файл
        if os.path.exists(tmp_model_path):
            os.remove(tmp_model_path)

        return True
    except Exception as e:
        logger.error(f"Ошибка при сохранении модели в W&B: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Store model in W&B Model Registry')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model file')
    parser.add_argument('--num_classes', type=int, default=102, help='Number of classes in the model')
    parser.add_argument('--project_name', type=str, default="diseases-classification", help='W&B project name')

    args = parser.parse_args()

    # Инициализация W&B
    if not init_wandb(args.project_name):
        logger.error("Не удалось инициализировать W&B. Завершение.")
        return

    # Загрузка модели
    model = load_model(args.model_path, args.num_classes)
    if model is None:
        logger.error("Не удалось загрузить модель. Завершение.")
        wandb.finish()
        return

    # Сохранение модели в W&B
    metadata = {
        "framework": "pytorch",
        "date": datetime.now().strftime("%Y-%m-%d"),
        "model_type": "ResNet18",
        "dataset": "Agricultural Diseases",
        "num_classes": args.num_classes,
        "source_model_path": args.model_path
    }

    success = store_model_in_wandb(model, args.model_path, metadata)

    # Завершение W&B сессии
    wandb.finish()

    if success:
        logger.info("Модель успешно сохранена в W&B Model Registry")
    else:
        logger.error("Не удалось сохранить модель в W&B Model Registry")


if __name__ == "__main__":
    main()