#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import kfp
from kfp import dsl
from kfp import compiler
from kfp.dsl import component, Output, Input, Artifact, Dataset


# Определение компонентов пайплайна
@component
def load_inference_data(data: Output[Dataset]):
    """Компонент для загрузки данных для инференса"""
    import pandas as pd
    import numpy as np

    # Генерация синтетических данных
    n_samples = 100
    n_features = 10
    X = np.random.rand(n_samples, n_features)

    # Создание DataFrame
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(n_features)])

    # Сохранение данных
    df.to_csv(data.path, index=False)

    print(f"Данные для инференса сохранены в {data.path}")


@component
def load_model(model: Output[Artifact]):
    """Компонент для загрузки обученной модели"""
    import os
    import pickle
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier

    # Создание новой модели
    print(f"Создание новой модели...")

    # Генерация синтетических данных для обучения простой модели
    X = np.random.rand(1000, 10)
    y = np.random.randint(0, 2, 1000)

    # Обучение простой модели
    model_obj = RandomForestClassifier(n_estimators=10, random_state=42)
    model_obj.fit(X, y)

    # Сохранение модели
    with open(model.path, 'wb') as f:
        pickle.dump(model_obj, f)

    print(f"Новая модель создана и сохранена в {model.path}")


@component
def run_inference(
        data: Input[Dataset],
        model: Input[Artifact],
        predictions: Output[Dataset]
):
    """Компонент для выполнения инференса"""
    import pandas as pd
    import pickle

    # Загрузка данных
    df = pd.read_csv(data.path)

    # Загрузка модели
    with open(model.path, 'rb') as f:
        model_obj = pickle.load(f)

    # Выполнение предсказаний
    pred = model_obj.predict(df)

    # Добавление предсказаний в DataFrame
    result_df = df.copy()
    result_df['prediction'] = pred

    # Сохранение результатов
    result_df.to_csv(predictions.path, index=False)

    print(f"Результаты инференса сохранены в {predictions.path}")


@component
def save_results(
        predictions: Input[Dataset],
        summary: Output[Artifact]
):
    """Компонент для анализа и сохранения результатов инференса"""
    import pandas as pd
    import json
    from datetime import datetime

    # Загрузка результатов предсказаний
    df = pd.read_csv(predictions.path)

    # Анализ результатов
    n_samples = len(df)

    # Подсчет количества каждого класса предсказаний
    if 'prediction' in df.columns:
        prediction_counts = df['prediction'].value_counts().to_dict()
    else:
        prediction_counts = {}

    # Создание сводки
    summary_data = {
        'total_samples': n_samples,
        'prediction_distribution': prediction_counts,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    # Сохранение сводки
    with open(summary.path, 'w') as f:
        json.dump(summary_data, f)

    print(f"Сводка результатов сохранена в {summary.path}")


# Определение пайплайна
@dsl.pipeline(
    name="Inference Pipeline",
    description="Пайплайн для инференса модели машинного обучения"
)
def inference_pipeline():
    # Шаг 1: Загрузка данных для инференса
    load_data_task = load_inference_data()

    # Шаг 2: Загрузка модели
    load_model_task = load_model()

    # Шаг 3: Выполнение инференса
    inference_task = run_inference(
        data=load_data_task.outputs["data"],
        model=load_model_task.outputs["model"]
    )

    # Шаг 4: Сохранение и анализ результатов
    save_results(predictions=inference_task.outputs["predictions"])


# Компиляция пайплайна
if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=inference_pipeline,
        package_path="inference_pipeline.yaml"
    )
    print("Пайплайн успешно скомпилирован в файл inference_pipeline.yaml")
    print("Для визуализации пайплайна вы можете:")
    print("1. Установить и запустить локальный сервер Kubeflow Pipelines:")
    print("   - Запустите: minikube start --driver=docker")
    print(
        "   - Установите Kubeflow Pipelines: kubectl apply -k github.com/kubeflow/pipelines/manifests/kustomize/cluster-scoped-resources")
    print("   - Перенаправьте порт: kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8080:80")
    print("2. Откройте http://localhost:8080 в браузере")
    print("3. Загрузите файл inference_pipeline.yaml через UI")
    print("4. Или используйте Kubeflow Central Dashboard если у вас есть доступ")

    # Попытка загрузить пайплайн напрямую (если есть доступ к кластеру)
    try:
        client = kfp.Client()
        client.create_run_from_pipeline_package(
            "inference_pipeline.yaml",
            experiment_name="ML Inference"
        )
        print("Пайплайн успешно запущен на кластере!")
    except Exception as e:
        print("Не удалось автоматически запустить пайплайн на кластере:")
        print(f"Ошибка: {str(e)}")
        print("Вы можете загрузить YAML файл вручную через Kubeflow UI")