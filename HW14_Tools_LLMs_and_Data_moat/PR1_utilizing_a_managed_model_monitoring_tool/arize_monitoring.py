#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
# Оновлена версія для PR
Модуль для моніторингу ML моделі за допомогою Arize AI
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv
from typing import Dict, List, Optional, Tuple, Any
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from arize.pandas.logger import Client, Schema, ModelTypes
from arize.utils.types import Embedding, ModelTaskType, ModelPredictionType

# Завантаження змінних середовища
load_dotenv()

# Отримання ключів API
API_KEY = os.getenv("ARIZE_API_KEY")
SPACE_KEY = os.getenv("ARIZE_SPACE_KEY")

# Перевірка наявності ключів
if not API_KEY or not SPACE_KEY:
    raise ValueError("Необхідно встановити змінні середовища ARIZE_API_KEY та ARIZE_SPACE_KEY")

def generate_synthetic_crop_data(n_samples: int = 1000) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Генерує синтетичні дані для виявлення загроз сільськогосподарським культурам

    Args:
        n_samples: Кількість зразків для генерації

    Returns:
        Кортеж (features_df, labels) - ознаки та мітки класів
    """
    # Генеруємо синтетичні дані для класифікації
    X, y = make_classification(
        n_samples=n_samples,
        n_features=20,  # 20 ознак, що представляють різні характеристики рослин/загроз
        n_informative=10,
        n_redundant=5,
        n_classes=4,  # 4 класи: 0=здорова, 1=хвороба, 2=шкідники, 3=бур'яни
        random_state=42
    )

    # Створюємо DataFrame з осмисленими іменами колонок
    feature_names = [
        "green_level", "leaf_area", "moisture", "temperature", "humidity",
        "nitrogen_level", "phosphorus_level", "potassium_level", "ph_soil", "organic_matter",
        "leaf_spots", "stem_discoloration", "growth_rate", "texture_anomaly", "leaf_curl",
        "canopy_density", "root_structure", "chlorophyll_content", "pest_presence", "weed_density"
    ]

    features_df = pd.DataFrame(X, columns=feature_names)

    # Додаємо метадані
    today = datetime.now()
    features_df["timestamp"] = [today - timedelta(days=np.random.randint(0, 30)) for _ in range(n_samples)]
    features_df["region_id"] = np.random.choice(["central", "north", "south", "east", "west"], size=n_samples)
    features_df["crop_type"] = np.random.choice(["wheat", "corn", "potato", "sunflower", "barley"], size=n_samples)
    features_df["prediction_id"] = [f"pred_{i}" for i in range(n_samples)]

    # Мітки класів
    labels = pd.Series(y, name="threat_class")

    return features_df, labels

def train_model(X: pd.DataFrame, y: pd.Series) -> RandomForestClassifier:
    """
    Навчає модель для виявлення загроз

    Args:
        X: DataFrame з ознаками
        y: Серія з мітками класів

    Returns:
        Навчена модель
    """
    # Навчаємо простий RandomForest класифікатор
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

def generate_embeddings(X: pd.DataFrame, n_dimensions: int = 10) -> List[List[float]]:
    """
    Генерує ембедінги для текстових/візуальних даних (симуляція)

    Args:
        X: DataFrame з ознаками
        n_dimensions: Розмірність ембедінгів

    Returns:
        Список ембедінгів
    """
    # Симулюємо ембедінги (в реальному випадку це були б ембедінги з моделі комп'ютерного зору або NLP)
    embeddings = []
    for _, row in X.iterrows():
        # Генеруємо випадковий вектор для симуляції ембедінгу зображення
        embedding = np.random.randn(n_dimensions).tolist()
        embeddings.append(embedding)
    return embeddings

def monitor_with_arize(features: pd.DataFrame, predictions: pd.Series, 
                      actuals: Optional[pd.Series] = None, 
                      embeddings: Optional[List[List[float]]] = None,
                      model_id: str = "crop_threat_detection_model",
                      model_version: str = "1.0.0") -> None:
    """
    Відправляє дані до Arize для моніторингу моделі

    Args:
        features: DataFrame з ознаками
        predictions: Серія з прогнозами моделі
        actuals: Серія з фактичними значеннями (ground truth)
        embeddings: Список ембедінгів для візуальних даних
        model_id: Ідентифікатор моделі
        model_version: Версія моделі
    """
    # Ініціалізуємо клієнт Arize
    arize_client = Client(space_key=SPACE_KEY, api_key=API_KEY)

    # Створюємо DataFrame для логування
    logging_df = features.copy()

    # Додаємо стовпець із прогнозами
    logging_df["prediction"] = predictions

    # Додаємо стовпець із фактичними значеннями, якщо вони доступні
    if actuals is not None:
        logging_df["actual"] = actuals

    # Додаємо ембедінги, якщо вони доступні
    embedding_list = None
    if embeddings is not None:
        embedding_list = [Embedding(vector=emb) for emb in embeddings]

    # Визначаємо схему для логування
    schema = Schema(
        prediction_id_column_name="prediction_id",
        timestamp_column_name="timestamp",
        prediction_label_column_name="prediction",
        actual_label_column_name="actual" if actuals is not None else None,
        feature_column_names=features.columns.tolist(),
        embedding_feature_column_names=None,  # В реальному випадку можна додати ембедінги
        model_type=ModelTypes.MULTICLASS_CLASSIFICATION,
        model_task=ModelTaskType.CLASSIFICATION,
        prediction_type=ModelPredictionType.CATEGORICAL,
        tag_column_names=["region_id", "crop_type"]
    )

    # Логуємо дані до Arize
    response = arize_client.log(
        dataframe=logging_df,
        schema=schema,
        model_id=model_id,
        model_version=model_version,
        embeddings=embedding_list
    )

    if response.status_code == 200:
        print(f"Успішно відправлено {len(logging_df)} записів до Arize")
        print(f"URL моніторингу: https://app.arize.com/models/{model_id}")
    else:
        print(f"Помилка при відправці даних до Arize: {response.text}")

def main():
    """
    Головна функція, що демонструє робочий процес моніторингу моделі за допомогою Arize
    """
    print("Генерація синтетичних даних для виявлення загроз сільськогосподарським культурам...")
    features, labels = generate_synthetic_crop_data(1000)

    # Розділяємо дані на навчальні та тестові
    X_train, X_test, y_train, y_test = train_test_split(
        features.drop(["timestamp", "region_id", "crop_type", "prediction_id"], axis=1), 
        labels, 
        test_size=0.3, 
        random_state=42
    )

    print("Навчання моделі...")
    model = train_model(X_train, y_train)

    # Генеруємо прогнози
    print("Генерація прогнозів...")
    test_features = features.iloc[y_test.index]
    predictions = pd.Series(model.predict(X_test), index=y_test.index, name="prediction")
    prediction_probs = model.predict_proba(X_test)

    # Генеруємо ембедінги (симуляція)
    print("Генерація ембедінгів...")
    embeddings = generate_embeddings(test_features)

    # Відправляємо дані до Arize для моніторингу
    print("Відправка даних до Arize для моніторингу...")
    monitor_with_arize(
        features=test_features,
        predictions=predictions,
        actuals=y_test,
        embeddings=embeddings,
        model_id="crop_threat_detection_model",
        model_version="1.0.0"
    )

    print("Готово!")

if __name__ == "__main__":
    main()
