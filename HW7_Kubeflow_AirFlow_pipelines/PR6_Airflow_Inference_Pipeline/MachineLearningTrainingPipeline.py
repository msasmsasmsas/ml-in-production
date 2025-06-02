#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import kfp
from kfp import dsl
from kfp import compiler
from kfp.dsl import component

# Определение компонентов пайплайна
@component
def load_data(data_path: str) -> str:
    """Компонент для загрузки данных для обучения"""
    import pandas as pd
    import numpy as np
    import os
    
    # Генерация синтетических данных
    n_samples = 1000
    n_features = 10
    X = np.random.rand(n_samples, n_features)
    y = np.random.randint(0, 2, n_samples)
    
    # Создание DataFrame
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(n_features)])
    df['target'] = y
    
    # Создание директории, если она не существует
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    
    # Сохранение данных
    df.to_csv(data_path, index=False)
    
    print(f"Данные для обучения сохранены в {data_path}")
    return data_path

@component
def preprocess_data(data_path: str) -> dict:
    """Компонент для предобработки данных"""
    import pandas as pd
    from sklearn.model_selection import train_test_split
    
    # Загрузка данных
    df = pd.read_csv(data_path)
    
    # Разделение на признаки и целевую переменную
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Разделение на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Сохранение предобработанных данных
    train_path = data_path.replace('.csv', '_train.csv')
    test_path = data_path.replace('.csv', '_test.csv')
    
    pd.concat([X_train, y_train], axis=1).to_csv(train_path, index=False)
    pd.concat([X_test, y_test], axis=1).to_csv(test_path, index=False)
    
    print(f"Обучающие данные сохранены в {train_path}")
    print(f"Тестовые данные сохранены в {test_path}")
    
    return {'train_path': train_path, 'test_path': test_path}

@component
def train_model(train_data_path: str, model_path: str) -> str:
    """Компонент для обучения модели"""
    import pandas as pd
    import pickle
    from sklearn.ensemble import RandomForestClassifier
    import os
    
    # Загрузка обучающих данных
    train_df = pd.read_csv(train_data_path)
    X_train = train_df.drop('target', axis=1)
    y_train = train_df['target']
    
    # Обучение модели
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Создание директории, если она не существует
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Сохранение модели
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"Модель сохранена в {model_path}")
    return model_path

@component
def evaluate_model(model_path: str, test_data_path: str, metrics_path: str) -> dict:
    """Компонент для оценки качества модели"""
    import pandas as pd
    import pickle
    import json
    from sklearn.metrics import accuracy_score
    import os
    
    # Загрузка модели
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # Загрузка тестовых данных
    test_df = pd.read_csv(test_data_path)
    X_test = test_df.drop('target', axis=1)
    y_test = test_df['target']
    
    # Оценка модели
    y_pred = model.predict(X_test)
    accuracy = float(accuracy_score(y_test, y_pred))
    
    # Сохранение метрик
    metrics = {'accuracy': accuracy}
    
    # Создание директории, если она не существует
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    
    # Сохранение метрик
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f)
    
    print(f"Точность модели: {accuracy}")
    print(f"Метрики сохранены в {metrics_path}")
    
    return metrics

# Определение пайплайна
@dsl.pipeline(
    name="Training Pipeline",
    description="Пайплайн для обучения модели машинного обучения"
)
def training_pipeline(
    data_path: str = "/tmp/data.csv",
    model_path: str = "/tmp/model.pkl",
    metrics_path: str = "/tmp/metrics.json"
):
    # Шаг 1: Загрузка данных
    load_data_task = load_data(data_path=data_path)
    
    # Шаг 2: Предобработка данных
    preprocess_task = preprocess_data(data_path=load_data_task.output)
    
    # Шаг 3: Обучение модели
    train_task = train_model(
        train_data_path=preprocess_task.output['train_path'],
        model_path=model_path
    )
    
    # Шаг 4: Оценка модели
    evaluate_task = evaluate_model(
        model_path=train_task.output,
        test_data_path=preprocess_task.output['test_path'],
        metrics_path=metrics_path
    )

# Компиляция пайплайна
if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=training_pipeline,
        package_path="training_pipeline.yaml"
    )
    print("Пайплайн успешно скомпилирован в файл training_pipeline.yaml")
    print("Для запуска пайплайна на кластере выполните следующие действия:")
    print("1. Загрузите YAML-файл через Kubeflow UI")
    print("2. Создайте запуск (run) с нужными параметрами")
    
    # Попытка загрузить пайплайн напрямую (если есть доступ к кластеру)
    try:
        client = kfp.Client()
        client.create_run_from_pipeline_package(
            "training_pipeline.yaml",
            arguments={
                'data_path': '/tmp/data.csv',
                'model_path': '/tmp/model.pkl',
                'metrics_path': '/tmp/metrics.json'
            },
            experiment_name="ML Training"
        )
        print("Пайплайн успешно запущен на кластере!")
    except Exception as e:
        print(f"Не удалось автоматически запустить пайплайн на кластере: {str(e)}")
        print("Используйте ручную загрузку YAML-файла через Kubeflow UI")