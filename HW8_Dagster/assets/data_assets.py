# assets/data_assets.py

import os
import sys
import pandas as pd
import pickle
import json
from datetime import datetime

# Імпортуємо необхідні компоненти Dagster
from dagster import asset, MaterializeResult, AssetExecutionContext, MetadataValue

# Додаємо шлях до модуля crawler з HW5
sys.path.append("../HW5_Training_Experiments")

# Спробуємо імпортувати модуль crawler, якщо він існує
try:
    from HW5_Training_Experiments.crawler import WebCrawler
    CRAWLER_AVAILABLE = True
except ImportError:
    print("Модуль WebCrawler недоступний, використовуємо заглушку")
    CRAWLER_AVAILABLE = False
    
    # Створюємо заглушку для WebCrawler
    class WebCrawler:
        def __init__(self, base_url, output_path=None):
            self.base_url = base_url
            self.output_path = output_path or "data/crawled_data.csv"
            
        def crawl(self, max_pages=5):
            # Створюємо синтетичні дані
            import numpy as np
            
            # Створюємо синтетичні дані для демонстрації
            X = np.random.rand(100, 5)
            y = np.random.randint(0, 2, 100)
            
            # Створюємо DataFrame
            df = pd.DataFrame(
                np.column_stack([X, y]),
                columns=["feature_1", "feature_2", "feature_3", "feature_4", "feature_5", "label"]
            )
            
            # Зберігаємо дані
            os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
            df.to_csv(self.output_path, index=False)
            
            return {
                "url_count": 5,
                "data_count": len(df),
                "success_rate": 1.0
            }


@asset(
    description="Датасет, отриманий через веб-краулер",
    group_name="data",
    compute_kind="python"
)
def crawled_dataset(context: AssetExecutionContext) -> pd.DataFrame:
    """
    Отримує дані з веб-сайту за допомогою краулера і зберігає їх як датасет.
    """
    context.log.info("Початок збору даних через краулер")
    
    # Шлях для збереження даних
    data_path = "data/crawled_data.csv"
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    
    # Ініціалізуємо та запускаємо краулер
    crawler = WebCrawler(
        base_url="https://example.com",
        output_path=data_path
    )
    
    # Запускаємо процес краулінгу
    result = crawler.crawl(max_pages=10)
    
    # Завантажуємо отримані дані
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        
        # Додаємо метадані про датасет
        context.add_output_metadata({
            "row_count": len(df),
            "column_count": len(df.columns),
            "columns": MetadataValue.json(list(df.columns)),
            "preview": MetadataValue.md(df.head().to_markdown()),
            "crawled_urls": result.get("url_count", 0),
            "crawl_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        
        return df
    else:
        context.log.error(f"Не вдалося знайти файл даних: {data_path}")
        # Повертаємо пустий датафрейм у випадку помилки
        return pd.DataFrame()


@asset(
    description="Навчена модель машинного навчання",
    group_name="models",
    compute_kind="python",
    deps=["crawled_dataset"]
)
def trained_model(context: AssetExecutionContext, crawled_dataset: pd.DataFrame):
    """
    Навчає модель на датасеті і зберігає її з параметрами навчання.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    
    context.log.info("Початок навчання моделі")
    
    # Перевіряємо, чи є дані
    if crawled_dataset.empty:
        context.log.error("Отримано пустий датасет")
        return None
    
    # Розділяємо на ознаки та цільову змінну
    if 'label' not in crawled_dataset.columns:
        context.log.error("У датасеті відсутня колонка 'label'")
        return None
    
    X = crawled_dataset.drop('label', axis=1).values
    y = crawled_dataset['label'].values
    
    # Розділяємо на тренувальну та тестову вибірки
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Навчаємо модель
    model_params = {
        "n_estimators": 100,
        "max_depth": 10,
        "random_state": 42
    }
    model = RandomForestClassifier(**model_params)
    model.fit(X_train, y_train)
    
    # Оцінюємо модель
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred, average='weighted')),
        "precision": float(precision_score(y_test, y_pred, average='weighted')),
        "recall": float(recall_score(y_test, y_pred, average='weighted'))
    }
    
    # Зберігаємо модель
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "random_forest_model.pkl")
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    # Зберігаємо параметри та метрики
    params_path = os.path.join(models_dir, "model_params.json")
    model_info = {
        "model_type": "RandomForestClassifier",
        "params": model_params,
        "metrics": metrics,
        "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_rows": len(crawled_dataset),
        "dataset_columns": len(crawled_dataset.columns) - 1,  # Без цільової змінної
    }
    
    # Додаємо важливість ознак, якщо це можливо
    if hasattr(model, 'feature_importances_'):
        feature_names = crawled_dataset.drop('label', axis=1).columns
        feature_importance = {
            str(name): float(importance) 
            for name, importance in zip(feature_names, model.feature_importances_)
        }
        model_info["feature_importances"] = feature_importance
    
    with open(params_path, 'w') as f:
        json.dump(model_info, f, indent=4)
    
    # Додаємо метадані до активу
    context.add_output_metadata({
        "model_type": "RandomForestClassifier",
        "model_path": model_path,
        "params_path": params_path,
        "accuracy": metrics["accuracy"],
        "f1_score": metrics["f1"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "training_dataset_size": len(crawled_dataset),
        "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })
    
    # Якщо є важливість ознак, додаємо її до метаданих
    if hasattr(model, 'feature_importances_'):
        feature_names = crawled_dataset.drop('label', axis=1).columns
        feature_importance = {
            str(name): float(importance) 
            for name, importance in zip(feature_names, model.feature_importances_)
        }
        context.add_output_metadata({
            "feature_importance": MetadataValue.json(feature_importance)
        })
    
    return model
