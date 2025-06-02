#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import kfp
from kfp import dsl
from kfp import compiler
from kfp.dsl import component, Output, Input, Artifact, Dataset


# Определение компонентов пайплайна
@component
def load_data(data: Output[Dataset]):
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

    # Сохранение данных
    df.to_csv(data.path, index=False)

    print(f"Данные для обучения сохранены в {data.path}")


@component
def preprocess_data(
        data: Input[Dataset],
        train_data: Output[Dataset],
        test_data: Output[Dataset]
):
    """Компонент для предобработки данных"""
    import pandas as pd
    from sklearn.model_selection import train_test_split

    # Загрузка данных
    df = pd.read_csv(data.path)

    # Разделение на признаки и целевую переменную
    X = df.drop('target', axis=1)
    y = df['target']

    # Разделение на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Сохранение предобработанных данных
    pd.concat([X_train, y_train], axis=1).to_csv(train_data.path, index=False)
    pd.concat([X_test, y_test], axis=1).to_csv(test_data.path, index=False)

    print(f"Обучающие данные сохранены в {train_data.path}")
    print(f"Тестовые данные сохранены в {test_data.path}")


@component
def train_model(
        train_data: Input[Dataset],
        model: Output[Artifact]
):
    """Компонент для обучения модели"""
    import pandas as pd
    import pickle
    from sklearn.ensemble import RandomForestClassifier

    # Загрузка обучающих данных
    train_df = pd.read_csv(train_data.path)
    X_train = train_df.drop('target', axis=1)
    y_train = train_df['target']

    # Обучение модели
    model_obj = RandomForestClassifier(n_estimators=100, random_state=42)
    model_obj.fit(X_train, y_train)

    # Сохранение модели
    with open(model.path, 'wb') as f:
        pickle.dump(model_obj, f)

    print(f"Модель сохранена в {model.path}")


@component
def evaluate_model(
        model: Input[Artifact],
        test_data: Input[Dataset],
        metrics_file: Output[Artifact]
):
    """Компонент для оценки качества модели"""
    import pandas as pd
    import pickle
    import json
    from sklearn.metrics import accuracy_score

    # Загрузка модели
    with open(model.path, 'rb') as f:
        model_obj = pickle.load(f)

    # Загрузка тестовых данных
    test_df = pd.read_csv(test_data.path)
    X_test = test_df.drop('target', axis=1)
    y_test = test_df['target']

    # Оценка модели
    y_pred = model_obj.predict(X_test)
    accuracy = float(accuracy_score(y_test, y_pred))

    # Сохранение метрик
    metrics = {'accuracy': accuracy}

    # Сохранение метрик в файл
    with open(metrics_file.path, 'w') as f:
        json.dump(metrics, f)

    print(f"Точность модели: {accuracy}")
    print(f"Метрики сохранены в {metrics_file.path}")


# Определение пайплайна
@dsl.pipeline(
    name="Training Pipeline",
    description="Пайплайн для обучения модели машинного обучения"
)
def training_pipeline():
    # Шаг 1: Загрузка данных
    load_data_task = load_data()

    # Шаг 2: Предобработка данных
    preprocess_task = preprocess_data(data=load_data_task.outputs["data"])

    # Шаг 3: Обучение модели
    train_task = train_model(train_data=preprocess_task.outputs["train_data"])

    # Шаг 4: Оценка модели
    evaluate_model(
        model=train_task.outputs["model"],
        test_data=preprocess_task.outputs["test_data"]
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

    # Попытка загрузить пайплайн напрямую (только если есть доступ к кластеру)
    try:
        client = kfp.Client()
        client.create_run_from_pipeline_package(
            "training_pipeline.yaml",
            experiment_name="ML Training"
        )
        print("Пайплайн успешно запущен на кластере!")
    except Exception as e:
        print(f"Не удалось автоматически запустить пайплайн на кластере: {str(e)}")
        print("Используйте ручную загрузку YAML-файла через Kubeflow UI")