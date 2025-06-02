import unittest
import numpy as np
import os
import sys

# Добавляем путь к модулям из HW5_Training_Experiments
# sys.path.append(os.path.join(os.path.dirname(__file__), '../../HW5_Training_Experiments'))
# Изменяем путь к модулю
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))


class DataProcessingTests(unittest.TestCase):
    """Тесты для функций обработки данных"""

    def test_transform_correctness(self):
        """Тест корректности трансформации изображений"""
        from HW5_Training_Experiments.PR1.dataset import get_simple_transforms

        # Получаем трансформации для обучения
        transforms = get_simple_transforms(224, is_training=True)

        # Создаем тестовое изображение
        import torch
        from PIL import Image
        test_image = Image.new('RGB', (300, 300), color='red')

        # Применяем трансформации
        transformed = transforms(test_image)

        # Проверяем размерность
        self.assertEqual(transformed.shape[0], 3)  # RGB каналы
        self.assertEqual(transformed.shape[1], 224)  # Высота
        self.assertEqual(transformed.shape[2], 224)  # Ширина

    def test_dataset_loading(self):
        """Тест загрузки данных из датасета"""
        from HW5_Training_Experiments.PR1.dataset import FastAgriculturalRiskDataset

        # Создаем мини-датасет для тестов
        import pandas as pd
        import tempfile
        import shutil

        # Создаем временную директорию
        temp_dir = tempfile.mkdtemp()
        try:
            # Создаем тестовые изображения
            img_dir = os.path.join(temp_dir, "images")
            os.makedirs(img_dir, exist_ok=True)

            # Создаем тестовое изображение
            from PIL import Image
            test_img_path = os.path.join(img_dir, "test_img.jpg")
            Image.new('RGB', (100, 100), color='blue').save(test_img_path)

            # Создаем тестовый CSV файл
            csv_data = {
                "id": [1],
                "disease_name": ["test_disease"],
                "image_path": [test_img_path]
            }
            csv_path = os.path.join(temp_dir, "test_data.csv")
            pd.DataFrame(csv_data).to_csv(csv_path, index=False)

            # Проверяем загрузку данных
            dataset = FastAgriculturalRiskDataset(
                csv_file=csv_path,
                image_dir=img_dir,
                transform=None,
                risk_type="all"
            )

            # Проверяем, что датасет корректно загрузился
            self.assertEqual(len(dataset), 1)

        finally:
            # Удаляем временную директорию
            shutil.rmtree(temp_dir)

    def test_class_mapping(self):
        """Тест корректности маппинга классов"""
        from HW5_Training_Experiments.PR1.utils import create_class_mapping

        # Тестовые данные
        class_names = ["apple_disease", "banana_disease", "cherry_disease"]

        # Получаем маппинг
        class_to_idx = create_class_mapping(class_names)

        # Проверяем корректность маппинга
        self.assertEqual(len(class_to_idx), 3)
        self.assertEqual(class_to_idx["apple_disease"], 0)
        self.assertEqual(class_to_idx["banana_disease"], 1)
        self.assertEqual(class_to_idx["cherry_disease"], 2)


if __name__ == '__main__':
    unittest.main()