from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import json

# Определение путей
DATA_PATH = '/opt/airflow/data/train_data.csv'
MODEL_PATH = '/opt/airflow/data/model.pkl'
METRICS_PATH = '/opt/airflow/data/metrics.json'


# Функции для этапов пайплайна
def load_data(**kwargs):
    """Загрузка обучающих данных"""
    # Генерация синтетических данных для демонстрации
    n_samples = 1000
    n_features = 10
    X = np.random.rand(n_samples, n_features)
    y = np.random.randint(0, 2, n_samples)  # Бинарная классификация

    # Создание DataFrame
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(n_features)])
    df['target'] = y

    # Создание директории, если она не существует
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)

    # Сохранение данных
    df.to_csv(DATA_PATH, index=False)
    print(f"Данные сохранены в {DATA_PATH}")

    # Передача пути к данным
    return DATA_PATH


def preprocess_data(**kwargs):
    """Предобработка данных"""
    # Получение пути к данным из предыдущего шага
    data_path = kwargs['ti'].xcom_pull(task_ids='load_data')

    # Загрузка данных
    df = pd.read_csv(data_path)

    # Простая предобработка (можно расширить)
    # Разделение на признаки и целевую переменную
    X = df.drop('target', axis=1)
    y = df['target']

    # Разделение на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Сохранение предобработанных данных
    train_path = DATA_PATH.replace('.csv', '_train.csv')
    test_path = DATA_PATH.replace('.csv', '_test.csv')

    pd.concat([X_train, y_train], axis=1).to_csv(train_path, index=False)
    pd.concat([X_test, y_test], axis=1).to_csv(test_path, index=False)

    print(f"Обучающие данные сохранены в {train_path}")
    print(f"Тестовые данные сохранены в {test_path}")

    return {'train_path': train_path, 'test_path': test_path}


def train_model(**kwargs):
    """Обучение модели"""
    # Получение путей к предобработанным данным
    ti = kwargs['ti']
    data_paths = ti.xcom_pull(task_ids='preprocess_data')
    train_path = data_paths['train_path']

    # Загрузка обучающих данных
    train_df = pd.read_csv(train_path)
    X_train = train_df.drop('target', axis=1)
    y_train = train_df['target']

    # Обучение модели RandomForest
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Создание директории, если она не существует
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    # Сохранение модели
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)

    print(f"Модель сохранена в {MODEL_PATH}")
    return MODEL_PATH


def evaluate_model(**kwargs):
    """Оценка качества модели"""
    # Получение пути к модели и тестовым данным
    ti = kwargs['ti']
    model_path = ti.xcom_pull(task_ids='train_model')
    data_paths = ti.xcom_pull(task_ids='preprocess_data')
    test_path = data_paths['test_path']

    # Загрузка модели
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    # Загрузка тестовых данных
    test_df = pd.read_csv(test_path)
    X_test = test_df.drop('target', axis=1)
    y_test = test_df['target']

    # Оценка модели
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Сохранение метрик
    metrics = {'accuracy': accuracy}
    os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)

    with open(METRICS_PATH, 'w') as f:
        json.dump(metrics, f)

    print(f"Точность модели: {accuracy}")
    print(f"Метрики сохранены в {METRICS_PATH}")

    return metrics


# Определение DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2023, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'training_pipeline',
    default_args=default_args,
    description='Пайплайн обучения модели машинного обучения',
    schedule_interval=timedelta(days=1),
    catchup=False
)

# Определение задач
load_data_task = PythonOperator(
    task_id='load_data',
    python_callable=load_data,
    provide_context=True,
    dag=dag,
)

preprocess_data_task = PythonOperator(
    task_id='preprocess_data',
    python_callable=preprocess_data,
    provide_context=True,
    dag=dag,
)

train_model_task = PythonOperator(
    task_id='train_model',
    python_callable=train_model,
    provide_context=True,
    dag=dag,
)

evaluate_model_task = PythonOperator(
    task_id='evaluate_model',
    python_callable=evaluate_model,
    provide_context=True,
    dag=dag,
)

# Определение зависимостей задач
load_data_task >> preprocess_data_task >> train_model_task >> evaluate_model_task