#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
import sys
import os
import numpy as np
from typing import List, Dict, Any, Optional, Tuple

# Добавляем путь для импорта модулей из корневой директории проекта
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))


class DataProcessingTests(unittest.TestCase):
    """
    Тесты для функций обработки данных
    """
    
    def setUp(self):
        """Подготовка данных для тестов"""
        self.sample_data = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0]
        ])
        
    def test_normalization(self):
        """Тест функции нормализации данных"""
        from ML_Pipeline.data_preprocessing import normalize_data
        
        # Тест будет проходить, когда функция normalize_data будет реализована
        normalized = normalize_data(self.sample_data)
        
        # Проверка, что данные нормализованы (среднее 0, стандартное отклонение 1)
        self.assertAlmostEqual(np.mean(normalized), 0.0, delta=1e-10)
        self.assertAlmostEqual(np.std(normalized), 1.0, delta=1e-10)
    
    def test_missing_values_handling(self):
        """Тест обработки пропущенных значений"""
        from ML_Pipeline.data_preprocessing import handle_missing_values
        
        # Создаем данные с пропущенными значениями
        data_with_missing = np.array([
            [1.0, np.nan, 3.0],
            [4.0, 5.0, np.nan],
            [np.nan, 8.0, 9.0]
        ])
        
        # Тест будет проходить, когда функция handle_missing_values будет реализована
        cleaned_data = handle_missing_values(data_with_missing, strategy='mean')
        
        # Проверка, что нет пропущенных значений
        self.assertFalse(np.isnan(cleaned_data).any())


class ModelTests(unittest.TestCase):
    """
    Тесты для моделей машинного обучения
    """
    
    def setUp(self):
        """Подготовка данных для тестов"""
        # Создаем простые тестовые данные
        np.random.seed(42)
        self.X_train = np.random.rand(100, 10)
        self.y_train = np.random.randint(0, 2, 100)
        self.X_test = np.random.rand(20, 10)
        self.y_test = np.random.randint(0, 2, 20)
    
    def test_model_training(self):
        """Тест обучения модели"""
        from ML_Pipeline.models import train_model
        
        # Тест будет проходить, когда функция train_model будет реализована
        model = train_model(self.X_train, self.y_train, model_type='random_forest')
        
        # Проверка, что модель обучена и может делать предсказания
        predictions = model.predict(self.X_test)
        self.assertEqual(len(predictions), len(self.y_test))
    
    def test_model_evaluation(self):
        """Тест оценки качества модели"""
        from ML_Pipeline.evaluation import evaluate_model
        
        # Создаем фиктивные предсказания
        predictions = np.random.randint(0, 2, 20)
        
        # Тест будет проходить, когда функция evaluate_model будет реализована
        metrics = evaluate_model(self.y_test, predictions)
        
        # Проверка наличия основных метрик
        self.assertIn('accuracy', metrics)
        self.assertIn('precision', metrics)
        self.assertIn('recall', metrics)
        self.assertIn('f1', metrics)


if __name__ == '__main__':
    unittest.main()
