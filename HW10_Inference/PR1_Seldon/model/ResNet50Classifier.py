#!/usr/bin/env python3

"""
Модель класифікації зображень ResNet50 для розгортання Seldon Core.
"""

import torch
from torchvision import models, transforms
from typing import Dict, List, Union, Optional, Any
import numpy as np
import json
import os
from PIL import Image
import logging
import io

# Налаштування логування
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ResNet50Classifier:
    """Класифікатор зображень ResNet50 для Seldon Core."""

    def __init__(self):
        """
        Ініціалізація моделі та перетворень для попередньої обробки.
        """
        logger.info("Ініціалізація моделі класифікатора ResNet50")

        # Встановлення пристрою (GPU якщо доступний, інакше CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Використовується пристрій: {self.device}")

        # Завантаження попередньо навченої моделі ResNet50
        self.model = models.resnet50(pretrained=True)
        self.model.to(self.device)
        self.model.eval()

        # Визначення перетворень для попередньої обробки зображення
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        # Завантаження міток класів ImageNet
        self.class_labels = self._load_imagenet_labels()

        # Метадані моделі
        self.name = "resnet50"
        self.version = "1.0.0"

        logger.info("Модель ResNet50 успішно ініціалізована")

    def _load_imagenet_labels(self) -> Dict[int, str]:
        """
        Завантаження міток класів ImageNet з JSON файлу.
        Якщо файл не існує, повертає малий підмножину міток.

        Returns:
            Словник, що відображає індекси класів на назви класів
        """
        # Шлях до файлу міток класів ImageNet
        labels_path = os.path.join(os.path.dirname(__file__), "imagenet_classes.json")

        try:
            if os.path.exists(labels_path):
                with open(labels_path, "r") as f:
                    return json.load(f)
            else:
                # Створюємо мінімальний набір класів ImageNet
                logger.warning("Файл класів ImageNet не знайдено. Створення малої підмножини.")
                # Створюємо файл з мінімальним набором класів
                fallback_labels = {
                    "0": "tench",
                    "1": "goldfish",
                    "2": "great white shark",
                    "3": "tiger shark",
                    "4": "hammerhead shark"
                }
                # Створюємо директорію якщо вона не існує
                os.makedirs(os.path.dirname(labels_path), exist_ok=True)
                # Зберігаємо файл
                with open(labels_path, "w") as f:
                    json.dump(fallback_labels, f)
                return fallback_labels
        except Exception as e:
            logger.error(f"Помилка завантаження міток ImageNet: {e}")
            return {"0": "unknown"}

    def predict(self, X: Union[np.ndarray, bytes, str, Image.Image], 
               feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Виконує прогнозування на вхідних даних.

        Args:
            X: Вхідні дані - можуть бути байти зображення, шлях до файлу, масив numpy або PIL Image
            feature_names: Назви ознак (не використовуються для класифікації зображень)

        Returns:
            Словник, що містить прогнозування
        """
        try:
            # Перетворення вхідних даних на PIL Image
            image = self._convert_to_image(X)

            # Попередня обробка зображення
            input_tensor = self.preprocess(image)
            input_batch = input_tensor.unsqueeze(0).to(self.device)

            # Виконання інференсу
            with torch.no_grad():
                output = self.model(input_batch)
                probabilities = torch.nn.functional.softmax(output[0], dim=0)

            # Отримання топ-5 прогнозувань
            top_probs, top_indices = torch.topk(probabilities, 5)

            # Форматування прогнозувань
            predictions = []
            for i, (prob, idx) in enumerate(zip(top_probs.cpu().numpy(), top_indices.cpu().numpy())):
                class_id = int(idx)
                class_name = self.class_labels.get(str(class_id), f"Невідомий клас {class_id}")
                predictions.append({
                    "rank": i + 1,
                    "class_id": class_id,
                    "class_name": class_name,
                    "probability": float(prob)
                })

            # Повернення форматованої відповіді
            return {
                "predictions": predictions,
                "model_name": self.name,
                "model_version": self.version
            }

        except Exception as e:
            logger.error(f"Помилка прогнозування: {e}")
            return {"error": str(e)}

    def _convert_to_image(self, X: Union[np.ndarray, bytes, str, Image.Image]) -> Image.Image:
        """
        Конвертує різні типи вхідних даних у PIL Image.

        Args:
            X: Вхідні дані в різних форматах

        Returns:
            Об'єкт PIL Image
        """
        # Якщо вхідні дані вже є PIL Image
        if isinstance(X, Image.Image):
            return X

        # Якщо вхідні дані є шляхом до файлу
        elif isinstance(X, str) and os.path.isfile(X):
            return Image.open(X).convert("RGB")

        # Якщо вхідні дані є байтами зображення
        elif isinstance(X, bytes):
            return Image.open(io.BytesIO(X)).convert("RGB")

        # Якщо вхідні дані є масивом numpy
        elif isinstance(X, np.ndarray):
            # Обробка різних форм масиву
            if X.ndim == 3 and X.shape[2] == 3:  # Формат (H, W, C)
                return Image.fromarray(X.astype(np.uint8))
            elif X.ndim == 3 and X.shape[0] == 3:  # Формат (C, H, W)
                X = X.transpose(1, 2, 0)
                return Image.fromarray(X.astype(np.uint8))
            elif X.ndim == 4 and X.shape[0] == 1:  # Формат (B, C, H, W) з розміром пакету 1
                X = X[0].transpose(1, 2, 0)
                return Image.fromarray(X.astype(np.uint8))
            else:
                raise ValueError(f"Непідтримувана форма масиву numpy: {X.shape}")

        # Якщо вхідні дані є списком, спробуємо спочатку конвертувати в масив numpy
        elif isinstance(X, list):
            return self._convert_to_image(np.array(X))

        # Непідтримуваний тип вхідних даних
        else:
            raise TypeError(f"Непідтримуваний тип вхідних даних: {type(X)}")

# Тестова функція для незалежного запуску моделі
def main():
    """Тестова функція для незалежного запуску моделі."""
    import argparse

    parser = argparse.ArgumentParser(description="Тестування класифікатора ResNet50")
    parser.add_argument("--image", type=str, required=True, help="Шлях до вхідного зображення")
    args = parser.parse_args()

    # Ініціалізація моделі
    classifier = ResNet50Classifier()

    # Виконання прогнозування
    result = classifier.predict(args.image)

    # Виведення результатів
    print("\nРезультати прогнозування:")
    print(f"Модель: {result['model_name']} (версія {result['model_version']})")
    print("\nТоп прогнозувань:")

    for pred in result["predictions"]:
        print(f"{pred['rank']}. {pred['class_name']}: {pred['probability']:.4f}")

if __name__ == "__main__":
    main()
