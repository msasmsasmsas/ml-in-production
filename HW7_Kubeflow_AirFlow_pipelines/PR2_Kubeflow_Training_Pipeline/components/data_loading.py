from kfp.components import create_component_from_func

@create_component_from_func
def load_training_data(
    data_path: str,
    output_path: str,
    train_test_split_ratio: float = 0.2,
    random_seed: int = 42
):
    """
    Компонент для загрузки и подготовки данных для обучения модели.
    
    Args:
        data_path: Путь к исходным данным
        output_path: Путь для сохранения подготовленных данных
        train_test_split_ratio: Доля тестовых данных
        random_seed: Случайное число для воспроизводимости
    
    Returns:
        Пути к обучающему и тестовому наборам данных
    """
    import os
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    
    # Создаем директорию для выходных данных
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Загружаем данные
    print(f"Загрузка данных из {data_path}")
    if data_path.endswith('.csv'):
        data = pd.read_csv(data_path)
    elif data_path.endswith('.parquet'):
        data = pd.read_parquet(data_path)
    else:
        raise ValueError(f"Неподдерживаемый формат файла: {data_path}")
    
    print(f"Загружено {len(data)} записей")
    
    # Разделяем данные на обучающую и тестовую выборки
    train_df, test_df = train_test_split(
        data, 
        test_size=train_test_split_ratio, 
        random_state=random_seed
    )
    
    print(f"Разделение данных: {len(train_df)} для обучения, {len(test_df)} для тестирования")
    
    # Сохраняем данные
    train_path = f"{output_path}/train.csv"
    test_path = f"{output_path}/test.csv"
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"Данные сохранены в {train_path} и {test_path}")
    
    return {
        'train_data_path': train_path,
        'test_data_path': test_path
    }
