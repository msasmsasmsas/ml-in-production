import unittest
import os
import sys
import tempfile
import shutil
import torch
import pandas as pd
from PIL import Image

# Добавляем путь к модулям из HW5_Training_Experiments
#sys.path.append(os.path.join(os.path.dirname(__file__), '../../HW5_Training_Experiments'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))


class PipelineIntegrationTests(unittest.TestCase):
    """Интеграционные тесты для полного пайплайна обучения"""

    def setUp(self):
        """Настройка тестового окружения"""
        # Создаем временную директорию для тестов
        self.temp_dir = tempfile.mkdtemp()
        self.img_dir = os.path.join(self.temp_dir, "images")
        os.makedirs(self.img_dir, exist_ok=True)

        # Создаем тестовые изображения для двух классов
        self.create_test_images()

        # Создаем тестовый CSV файл
        self.create_test_csv()

    def tearDown(self):
        """Очистка после тестов"""
        # Удаляем временную директорию
        shutil.rmtree(self.temp_dir)

    def create_test_images(self):
        """Создание тестовых изображений"""
        # Создаем 3 изображения для класса 1
        for i in range(3):
            img_path = os.path.join(self.img_dir, f"class1_img{i}.jpg")
            Image.new('RGB', (100, 100), color='red').save(img_path)

        # Создаем 3 изображения для класса 2
        for i in range(3):
            img_path = os.path.join(self.img_dir, f"class2_img{i}.jpg")
            Image.new('RGB', (100, 100), color='blue').save(img_path)

    def create_test_csv(self):
        """Создание тестового CSV файла с метаданными"""
        # Данные для CSV
        data = {
            "id": list(range(6)),
            "disease_name": ["class1"] * 3 + ["class2"] * 3,
            "image_path": [
                              os.path.join(self.img_dir, f"class1_img{i}.jpg") for i in range(3)
                          ] + [
                              os.path.join(self.img_dir, f"class2_img{i}.jpg") for i in range(3)
                          ]
        }

        # Сохраняем CSV
        self.csv_path = os.path.join(self.temp_dir, "test_data.csv")
        pd.DataFrame(data).to_csv(self.csv_path, index=False)

    def test_end_to_end_pipeline(self):
        """
        Тест для проверки полного пайплайна:
        от загрузки данных до обучения модели
        """
        try:
            # Импортируем функции из модуля обучения
            from HW5_Training_Experiments.PR2.model import create_model
            from HW5_Training_Experiments.PR1.dataset import get_simple_transforms
            from HW5_Training_Experiments.PR1.dataset import FastAgriculturalRiskDataset
            from torch.utils.data import DataLoader

            # 1. Создаем датасет
            transform = get_simple_transforms(224, is_training=True)
            dataset = FastAgriculturalRiskDataset(
                csv_file=self.csv_path,
                image_dir=self.img_dir,
                transform=transform,
                risk_type="all"
            )

            # 2. Создаем DataLoader
            dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

            # 3. Создаем модель
            model = create_model(num_classes=2)

            # 4. Проверяем, что модель может обрабатывать батч
            batch_x, batch_y = next(iter(dataloader))
            outputs = model(batch_x)

            # 5. Проверяем размерность выходных данных
            self.assertEqual(outputs.shape[0], batch_x.shape[0])  # Размер батча
            self.assertEqual(outputs.shape[1], 2)  # Число классов

            # 6. Проверяем один шаг обучения
            criterion = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

            # Forward
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)

            # Backward + optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Проверяем, что loss имеет значение
            self.assertIsNotNone(loss.item())

        except Exception as e:
            self.fail(f"Ошибка в пайплайне обучения: {str(e)}")


if __name__ == '__main__':
    unittest.main()