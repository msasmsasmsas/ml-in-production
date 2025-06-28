from kfp.components import create_component_from_func

@create_component_from_func
def load_training_data(
    data_path: str,
    output_path: str,
    train_test_split_ratio: float = 0.2,
    random_seed: int = 42
):
    """
    Компонент для завантаження та підготовки даних для навчання моделі.

    Args:
        data_path: Шлях до вихідних даних
        output_path: Шлях для збереження підготовлених даних
        train_test_split_ratio: Частка тестових даних
        random_seed: Випадкове число для відтворюваності

    Returns:
        Шляхи до тренувального та тестового наборів даних
    """
    import os
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split

    # Створюємо директорію для вихідних даних
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Завантажуємо дані
    print(f"Завантаження даних з {data_path}")
    if data_path.endswith('.csv'):
        data = pd.read_csv(data_path)
    elif data_path.endswith('.parquet'):
        data = pd.read_parquet(data_path)
    else:
        raise ValueError(f"Непідтримуваний формат файлу: {data_path}")

    print(f"Завантажено {len(data)} записів")

    # Розділяємо дані на тренувальну та тестову вибірки
    train_df, test_df = train_test_split(
        data, 
        test_size=train_test_split_ratio, 
        random_state=random_seed
    )

    print(f"Розділення даних: {len(train_df)} для навчання, {len(test_df)} для тестування")

    # Зберігаємо дані
    train_path = f"{output_path}/train.csv"
    test_path = f"{output_path}/test.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Дані збережено в {train_path} та {test_path}")
    
    return {
        'train_data_path': train_path,
        'test_data_path': test_path
    }
