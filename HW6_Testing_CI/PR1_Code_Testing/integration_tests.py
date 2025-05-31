#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
import sys
import os
import tempfile
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple

# Добавляем путь для импорта модулей из корневой директории проекта
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))


class PipelineIntegrationTests(unittest.TestCase):
    """
    Интеграционные тесты для проверки работы ML пайплайна
    """
    
    def setUp(self):
        """Подготовка данных для тестов"""
        # Создаем временную директорию для сохранения артефактов
        self.test_dir = tempfile.TemporaryDirectory()
        
        # Создаем тестовые данные
        np.random.seed(42)
        n_samples = 100
        n_features = 5
        
        # Создаем синтетические данные
        X = np.random.rand(n_samples, n_features)
        y = np.random.randint(0, 2, n_samples)
        
        # Создаем DataFrame
        feature_names = [f'feature_{i}' for i in range(n_features)]
        self.test_data = pd.DataFrame(X, columns=feature_names)
        self.test_data['target'] = y
        
        # Сохраняем данные во временный файл
        self.data_path = os.path.join(self.test_dir.name, 'test_data.csv')
        self.test_data.to_csv(self.data_path, index=False)
        
        # Путь для сохранения модели
        self.model_path = os.path.join(self.test_dir.name, 'model.pkl')
    
    def tearDown(self):
        """Очистка после тестов"""
        self.test_dir.cleanup()
    
    def test_end_to_end_pipeline(self):
        """
        Тест для проверки полного пайплайна: 
        от загрузки данных до оценки модели
        """
        from ML_Pipeline.data_loading import load_data
        from ML_Pipeline.data_preprocessing import preprocess_data
        from ML_Pipeline.feature_engineering import engineer_features
        from ML_Pipeline.models import train_model, save_model, load_model
        from ML_Pipeline.evaluation import evaluate_model
        
        # Загрузка данных
        data = load_data(self.data_path)
        self.assertEqual(len(data), len(self.test_data))
        
        # Предобработка данных
        X, y = preprocess_data(data, target_column='target')
        self.assertEqual(X.shape[0], len(self.test_data))
        self.assertEqual(y.shape[0], len(self.test_data))
        
        # Генерация признаков
        X_featured = engineer_features(X)
        
        # Разделение на обучающую и тестовую выборки
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X_featured, y, test_size=0.2, random_state=42
        )
        
        # Обучение модели
        model = train_model(X_train, y_train, model_type='random_forest')
        
        # Сохранение модели
        save_model(model, self.model_path)
        self.assertTrue(os.path.exists(self.model_path))
        
        # Загрузка модели
        loaded_model = load_model(self.model_path)
        
        # Предсказание
        y_pred = loaded_model.predict(X_test)
        self.assertEqual(len(y_pred), len(y_test))
        
        # Оценка модели
        metrics = evaluate_model(y_test, y_pred)
        self.assertIn('accuracy', metrics)
        self.assertIn('precision', metrics)
        self.assertIn('recall', metrics)
        self.assertIn('f1', metrics)


if __name__ == '__main__':
    unittest.main()
