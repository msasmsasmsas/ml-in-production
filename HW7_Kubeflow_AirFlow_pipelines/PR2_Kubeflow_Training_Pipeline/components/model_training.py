from kfp.components import create_component_from_func

@create_component_from_func
def train_model(
    train_data_path: str,
    test_data_path: str,
    model_output_path: str,
    target_column: str,
    hyperparameters: dict = None
):
    """
    Компонент для обучения модели машинного обучения.
    
    Args:
        train_data_path: Путь к обучающим данным
        test_data_path: Путь к тестовым данным
        model_output_path: Путь для сохранения обученной модели
        target_column: Имя целевой колонки
        hyperparameters: Словарь с гиперпараметрами модели
        
    Returns:
        Метрики модели и путь к сохраненной модели
    """
    import os
    import pandas as pd
    import numpy as np
    import pickle
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    # Создаем директорию для выходных данных
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    
    # Загружаем данные
    print(f"Загрузка данных из {train_data_path} и {test_data_path}")
    train_df = pd.read_csv(train_data_path)
    test_df = pd.read_csv(test_data_path)
    
    # Подготовка данных
    X_train = train_df.drop(columns=[target_column])
    y_train = train_df[target_column]
    X_test = test_df.drop(columns=[target_column])
    y_test = test_df[target_column]
    
    # Устанавливаем параметры модели
    params = {
        'n_estimators': 100,
        'max_depth': 10,
        'random_state': 42
    }
    
    # Обновляем параметры, если они предоставлены
    if hyperparameters:
        params.update(hyperparameters)
    
    print(f"Обучение модели с параметрами: {params}")
    
    # Создаем и обучаем модель
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    
    # Оцениваем модель
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"Метрики модели:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    
    # Сохраняем модель
    with open(model_output_path, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"Модель сохранена в {model_output_path}")
    
    # Возвращаем метрики и путь к модели
    return {
        'model_path': model_output_path,
        'metrics': {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1)
        }
    }
