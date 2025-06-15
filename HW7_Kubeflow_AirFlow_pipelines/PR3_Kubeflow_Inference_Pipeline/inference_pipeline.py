#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import kfp
from kfp import dsl
from kfp import compiler
from kfp.dsl import component, Output, Input, Artifact, Dataset


# Визначення компонентів пайплайну
@component
def load_inference_data(data: Output[Dataset]):
    """Компонент для завантаження даних для інференсу"""
    import pandas as pd
    import numpy as np

    # Генерація синтетичних даних
    n_samples = 100
    n_features = 10
    X = np.random.rand(n_samples, n_features)

    # Створення DataFrame
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(n_features)])

    # Збереження даних
    df.to_csv(data.path, index=False)

    print(f"Дані для інференсу збережено в {data.path}")


@component
def load_model(model: Output[Artifact]):
    """Компонент для завантаження навченої моделі"""
    import os
    import pickle
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier

    # Створення нової моделі
    print(f"Створення нової моделі...")

    # Генерація синтетичних даних для навчання простої моделі
    X = np.random.rand(1000, 10)
    y = np.random.randint(0, 2, 1000)

    # Навчання простої моделі
    model_obj = RandomForestClassifier(n_estimators=10, random_state=42)
    model_obj.fit(X, y)

    # Збереження моделі
    with open(model.path, 'wb') as f:
        pickle.dump(model_obj, f)

    print(f"Нова модель створена та збережена в {model.path}")


@component
def run_inference(
        data: Input[Dataset],
        model: Input[Artifact],
        predictions: Output[Dataset]
):
    """Компонент для виконання інференсу"""
    import pandas as pd
    import pickle

    # Завантаження даних
    df = pd.read_csv(data.path)

    # Завантаження моделі
    with open(model.path, 'rb') as f:
        model_obj = pickle.load(f)

    # Виконання прогнозувань
    pred = model_obj.predict(df)

    # Додавання прогнозувань до DataFrame
    result_df = df.copy()
    result_df['prediction'] = pred

    # Збереження результатів
    result_df.to_csv(predictions.path, index=False)

    print(f"Результати інференсу збережено в {predictions.path}")


@component
def save_results(
        predictions: Input[Dataset],
        summary: Output[Artifact]
):
    """Компонент для аналізу та збереження результатів інференсу"""
    import pandas as pd
    import json
    from datetime import datetime

    # Завантаження результатів прогнозувань
    df = pd.read_csv(predictions.path)

    # Аналіз результатів
    n_samples = len(df)

    # Підрахунок кількості кожного класу прогнозувань
    if 'prediction' in df.columns:
        prediction_counts = df['prediction'].value_counts().to_dict()
    else:
        prediction_counts = {}

    # Створення підсумку
    summary_data = {
        'total_samples': n_samples,
        'prediction_distribution': prediction_counts,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    # Збереження підсумку
    with open(summary.path, 'w') as f:
        json.dump(summary_data, f)

    print(f"Підсумок результатів збережено в {summary.path}")


# Визначення пайплайну
@dsl.pipeline(
    name="Inference Pipeline",
    description="Пайплайн для інференсу моделі машинного навчання"
)
def inference_pipeline():
    # Крок 1: Завантаження даних для інференсу
    load_data_task = load_inference_data()

    # Крок 2: Завантаження моделі
    load_model_task = load_model()

    # Крок 3: Виконання інференсу
    inference_task = run_inference(
        data=load_data_task.outputs["data"],
        model=load_model_task.outputs["model"]
    )

    # Крок 4: Збереження та аналіз результатів
    save_results(predictions=inference_task.outputs["predictions"])


# Компіляція пайплайну
if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=inference_pipeline,
        package_path="inference_pipeline.yaml"
    )
    print("Пайплайн успішно скомпільовано в файл inference_pipeline.yaml")
    print("Для візуалізації пайплайну ви можете:")
    print("1. Встановити та запустити локальний сервер Kubeflow Pipelines:")
    print("   - Запустіть: minikube start --driver=docker")
    print(
        "   - Встановіть Kubeflow Pipelines: kubectl apply -k github.com/kubeflow/pipelines/manifests/kustomize/cluster-scoped-resources")
    print("   - Перенаправте порт: kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8080:80")
    print("2. Відкрийте http://localhost:8080 у браузері")
    print("3. Завантажте файл inference_pipeline.yaml через UI")
    print("4. Або використовуйте Kubeflow Central Dashboard якщо у вас є доступ")

    # Спроба завантажити пайплайн напряму (якщо є доступ до кластера)
    try:
        client = kfp.Client()
        client.create_run_from_pipeline_package(
            "inference_pipeline.yaml",
            experiment_name="ML Inference"
        )
        print("Пайплайн успішно запущено на кластері!")
    except Exception as e:
        print("Не вдалося автоматично запустити пайплайн на кластері:")
        print(f"Помилка: {str(e)}")
        print("Ви можете завантажити YAML файл вручну через Kubeflow UI")