#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import logging
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# Импорт библиотек для валидации данных
try:
    import cleanlab
    from cleanlab.dataset import health_summary
    from cleanlab.classification import CleanLearning
    from cleanlab.filter import find_label_issues
    CLEANLAB_AVAILABLE = True
except ImportError:
    CLEANLAB_AVAILABLE = False
    logging.warning("Cleanlab не установлен. Некоторые функции будут недоступны.")

try:
    import deepchecks
    from deepchecks.tabular import Dataset
    from deepchecks.tabular.checks import (
        DataDuplicates, 
        FeatureFeatureCorrelation,
        DatasetsSizeComparison,
        TrainTestFeatureDrift,
        TrainTestLabelDrift,
        MixedNulls,
        StringMismatch,
        ConflictingLabels
    )
    from deepchecks.tabular.suites import data_integrity, train_test_validation
    DEEPCHECKS_AVAILABLE = True
except ImportError:
    DEEPCHECKS_AVAILABLE = False
    logging.warning("Deepchecks не установлен. Некоторые функции будут недоступны.")

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("data_validation")

class DataValidator:
    """
    Класс для валидации и тестирования размеченных данных
    """
    def __init__(self, data_path: str, output_dir: str = 'validation_results'):
        self.data_path = data_path
        self.output_dir = output_dir
        self.data = None
        
        # Создание директории для результатов
        os.makedirs(output_dir, exist_ok=True)
        
        # Загрузка данных
        self._load_data()
    
    def _load_data(self):
        """
        Загрузка данных из файла CSV или JSON
        """
        file_ext = os.path.splitext(self.data_path)[1].lower()
        
        try:
            if file_ext == '.csv':
                self.data = pd.read_csv(self.data_path)
            elif file_ext == '.json':
                self.data = pd.read_json(self.data_path)
            else:
                raise ValueError(f"Неподдерживаемый формат файла: {file_ext}")
            
            logger.info(f"Загружено {len(self.data)} записей из {self.data_path}")
        
        except Exception as e:
            logger.error(f"Ошибка при загрузке данных: {e}")
            raise
    
    def basic_statistics(self) -> Dict[str, Any]:
        """
        Получение базовой статистики по данным
        """
        if self.data is None:
            return {}
        
        stats = {
            "total_records": len(self.data),
            "columns": list(self.data.columns),
            "missing_values": self.data.isnull().sum().to_dict(),
            "duplicates": self.data.duplicated().sum()
        }
        
        # Статистика по категориальным столбцам
        cat_columns = self.data.select_dtypes(include=['object', 'category']).columns
        stats["categorical_columns"] = {
            col: self.data[col].value_counts().to_dict() for col in cat_columns
        }
        
        # Статистика по числовым столбцам
        num_columns = self.data.select_dtypes(include=['int64', 'float64']).columns
        stats["numeric_columns"] = {
            col: {
                "min": self.data[col].min(),
                "max": self.data[col].max(),
                "mean": self.data[col].mean(),
                "median": self.data[col].median(),
                "std": self.data[col].std()
            } for col in num_columns
        }
        
        # Сохранение статистики в файл
        with open(os.path.join(self.output_dir, 'basic_statistics.json'), 'w') as f:
            json.dump(stats, f, indent=2, default=str)
        
        return stats
    
    def data_visualization(self):
        """
        Создание визуализаций для анализа данных
        """
        if self.data is None:
            return
        
        # Визуализация распределения категориальных переменных
        cat_columns = self.data.select_dtypes(include=['object', 'category']).columns
        for col in cat_columns:
            plt.figure(figsize=(12, 6))
            sns.countplot(y=col, data=self.data, order=self.data[col].value_counts().index)
            plt.title(f'Распределение значений в столбце {col}')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f'{col}_distribution.png'))
            plt.close()
        
        # Визуализация распределения числовых переменных
        num_columns = self.data.select_dtypes(include=['int64', 'float64']).columns
        for col in num_columns:
            plt.figure(figsize=(12, 6))
            sns.histplot(self.data[col], kde=True)
            plt.title(f'Распределение значений в столбце {col}')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f'{col}_distribution.png'))
            plt.close()
        
        # Матрица корреляции для числовых переменных
        if len(num_columns) > 1:
            plt.figure(figsize=(12, 10))
            correlation_matrix = self.data[num_columns].corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
            plt.title('Матрица корреляции')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'correlation_matrix.png'))
            plt.close()
        
        logger.info(f"Визуализации сохранены в директории {self.output_dir}")
    
    def validate_with_cleanlab(self, label_column: str):
        """
        Проверка качества разметки с помощью Cleanlab
        """
        if not CLEANLAB_AVAILABLE:
            logger.error("Cleanlab не установлен. Установите его с помощью: pip install cleanlab")
            return
        
        if self.data is None or label_column not in self.data.columns:
            logger.error(f"Столбец {label_column} не найден в данных")
            return
        
        try:
            # Получение базового отчета о здоровье данных
            health_report = health_summary(self.data, label_name=label_column)
            
            # Сохранение отчета
            with open(os.path.join(self.output_dir, 'cleanlab_health_report.json'), 'w') as f:
                json.dump(health_report, f, indent=2, default=str)
            
            # Визуализация отчета
            plt.figure(figsize=(12, 8))
            sns.barplot(x='issue_type', y='num_issues', data=pd.DataFrame(health_report))
            plt.title('Проблемы в данных (Cleanlab)')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'cleanlab_issues.png'))
            plt.close()
            
            logger.info(f"Отчет Cleanlab сохранен в {self.output_dir}")
        
        except Exception as e:
            logger.error(f"Ошибка при валидации с Cleanlab: {e}")
    
    def validate_with_deepchecks(self, label_column: str, split_column: Optional[str] = None):
        """
        Проверка качества данных с помощью Deepchecks
        """
        if not DEEPCHECKS_AVAILABLE:
            logger.error("Deepchecks не установлен. Установите его с помощью: pip install deepchecks")
            return
        
        if self.data is None or label_column not in self.data.columns:
            logger.error(f"Столбец {label_column} не найден в данных")
            return
        
        try:
            # Создание датасета Deepchecks
            cat_features = list(self.data.select_dtypes(include=['object', 'category']).columns)
            num_features = list(self.data.select_dtypes(include=['int64', 'float64']).columns)
            
            # Исключаем метки из списка признаков
            if label_column in cat_features:
                cat_features.remove(label_column)
            if label_column in num_features:
                num_features.remove(label_column)
            
            # Если указан столбец для разделения, создаем тренировочный и тестовый наборы
            if split_column and split_column in self.data.columns:
                train_data = self.data[self.data[split_column] == 'train']
                test_data = self.data[self.data[split_column] == 'test']
                
                train_ds = Dataset(
                    df=train_data,
                    label=label_column,
                    cat_features=cat_features
                )
                
                test_ds = Dataset(
                    df=test_data,
                    label=label_column,
                    cat_features=cat_features
                )
                
                # Запуск набора проверок для валидации тренировочного и тестового наборов
                suite = train_test_validation()
                result = suite.run(train_dataset=train_ds, test_dataset=test_ds)
                result.save_as_html(os.path.join(self.output_dir, 'deepchecks_train_test_validation.html'))
            
            # В любом случае проверяем целостность данных
            ds = Dataset(
                df=self.data,
                label=label_column,
                cat_features=cat_features
            )
            
            # Запуск набора проверок для целостности данных
            suite = data_integrity()
            result = suite.run(ds)
            result.save_as_html(os.path.join(self.output_dir, 'deepchecks_data_integrity.html'))
            
            logger.info(f"Отчеты Deepchecks сохранены в {self.output_dir}")
        
        except Exception as e:
            logger.error(f"Ошибка при валидации с Deepchecks: {e}")
    
    def find_mislabeled_samples(self, label_column: str, features_columns: List[str], 
                               model_type: str = 'random_forest', n_samples: int = 10):
        """
        Поиск потенциально неправильно размеченных примеров
        """
        if not CLEANLAB_AVAILABLE:
            logger.error("Cleanlab не установлен. Установите его с помощью: pip install cleanlab")
            return
        
        if self.data is None or label_column not in self.data.columns:
            logger.error(f"Столбец {label_column} не найден в данных")
            return
        
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.linear_model import LogisticRegression
            from sklearn.model_selection import cross_val_predict
            from sklearn.preprocessing import LabelEncoder, StandardScaler
            
            # Подготовка данных
            X = self.data[features_columns].copy()
            
            # Кодирование категориальных признаков
            for col in X.select_dtypes(include=['object', 'category']).columns:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
            
            # Нормализация числовых признаков
            scaler = StandardScaler()
            X[X.select_dtypes(include=['int64', 'float64']).columns] = scaler.fit_transform(
                X[X.select_dtypes(include=['int64', 'float64']).columns]
            )
            
            # Кодирование меток
            le = LabelEncoder()
            y = le.fit_transform(self.data[label_column].astype(str))
            
            # Выбор модели
            if model_type == 'random_forest':
                model = RandomForestClassifier(n_estimators=100, random_state=42)
            elif model_type == 'logistic_regression':
                model = LogisticRegression(max_iter=1000, random_state=42)
            else:
                raise ValueError(f"Неподдерживаемый тип модели: {model_type}")
            
            # Получение предсказаний вероятностей с перекрестной проверкой
            pred_probs = cross_val_predict(model, X, y, cv=5, method='predict_proba')
            
            # Поиск проблемных примеров
            label_issues = find_label_issues(y, pred_probs)
            
            # Индексы потенциально неправильно размеченных примеров
            issue_indices = np.where(label_issues)[0]
            
            # Создание датафрейма с проблемными примерами
            if len(issue_indices) > 0:
                issues_df = self.data.iloc[issue_indices].copy()
                issues_df['predicted_label'] = le.inverse_transform(pred_probs[issue_indices].argmax(axis=1))
                issues_df['confidence'] = pred_probs[issue_indices].max(axis=1)
                
                # Сохранение результатов
                issues_df.to_csv(os.path.join(self.output_dir, 'mislabeled_samples.csv'), index=False)
                
                # Выборка n_samples примеров для анализа
                sample_issues = issues_df.sample(min(n_samples, len(issues_df)))
                
                logger.info(f"Найдено {len(issues_df)} потенциально неправильно размеченных примеров")
                logger.info(f"Примеры сохранены в {os.path.join(self.output_dir, 'mislabeled_samples.csv')}")
                
                return issues_df
            else:
                logger.info("Не найдено потенциально неправильно размеченных примеров")
                return pd.DataFrame()
        
        except Exception as e:
            logger.error(f"Ошибка при поиске неправильно размеченных примеров: {e}")
            return pd.DataFrame()

def main():
    parser = argparse.ArgumentParser(description="Валидация и тестирование размеченных данных")
    parser.add_argument('--data_path', type=str, required=True, help='Путь к файлу с данными (CSV или JSON)')
    parser.add_argument('--output_dir', type=str, default='validation_results', help='Директория для сохранения результатов')
    parser.add_argument('--label_column', type=str, required=True, help='Имя столбца с метками классов')
    parser.add_argument('--split_column', type=str, help='Имя столбца для разделения на обучающую и тестовую выборки')
    parser.add_argument('--use_cleanlab', action='store_true', help='Использовать Cleanlab для проверки разметки')
    parser.add_argument('--use_deepchecks', action='store_true', help='Использовать Deepchecks для проверки данных')
    parser.add_argument('--find_mislabeled', action='store_true', help='Искать потенциально неправильно размеченные примеры')
    parser.add_argument('--features', type=str, nargs='+', help='Столбцы признаков для поиска неправильных меток')
    
    args = parser.parse_args()
    
    try:
        # Инициализация валидатора данных
        validator = DataValidator(args.data_path, args.output_dir)
        
        # Базовая статистика и визуализация
        validator.basic_statistics()
        validator.data_visualization()
        
        # Валидация с Cleanlab
        if args.use_cleanlab:
            validator.validate_with_cleanlab(args.label_column)
        
        # Валидация с Deepchecks
        if args.use_deepchecks:
            validator.validate_with_deepchecks(args.label_column, args.split_column)
        
        # Поиск неправильно размеченных примеров
        if args.find_mislabeled and args.features:
            validator.find_mislabeled_samples(args.label_column, args.features)
        
        logger.info(f"Валидация данных завершена. Результаты сохранены в {args.output_dir}")
    
    except Exception as e:
        logger.error(f"Ошибка при выполнении валидации: {e}")

if __name__ == "__main__":
    main()
