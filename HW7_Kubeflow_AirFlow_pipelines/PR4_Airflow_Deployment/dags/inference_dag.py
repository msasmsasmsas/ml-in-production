from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import pandas as pd
import numpy as np
import pickle
import os
import json

# Определение путей
MODEL_PATH = '/opt/airflow/data/model.pkl'
INFERENCE_DATA_PATH = '/opt/airflow/data/inference_data.csv'
PREDICTIONS_PATH = '/opt/airflow/data/predictions.csv'


# Функции для этапов пайплайна
def load_inference_data(**kwargs):
    """Загрузка данных для инференса"""
    # Генерация синтетических данных для демонстрации
    n_samples = 100
    n_features = 10
    X = np.random.rand(n_samples, n_features)

    # Создание DataFrame
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(n_features)])

    # Создание директории, если она не существует
    os.makedirs(os.path.dirname(INFERENCE_DATA_PATH), exist_ok=True)

    # Сохранение данных
    df.to_csv(INFERENCE_DATA_PATH, index=False)
    print(f"Данные для инференса сохранены в {INFERENCE_DATA_PATH}")

    return INFERENCE_DATA_PATH


def load_model(**kwargs):
    """Загрузка обученной модели"""
    # Проверка существования модели
    if not os.path.exists(MODEL_PATH):
        # Если модель не существует, обучаем новую
        print(f"Модель не найдена по пути {MODEL_PATH}. Создание новой модели...")

        # Генерация синтетических данных для обучения
        n_samples = 1000
        n_features = 10
        X = np.random.rand(n_samples, n_features)
        y = np.random.randint(0, 2, n_samples)

        # Обучение модели
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)

        # Создание директории, если она не существует
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

        # Сохранение модели
        with open(MODEL_PATH, 'wb') as f:
            pickle.dump(model, f)

        print(f"Новая модель сохранена в {MODEL_PATH}")
    else:
        print(f"Модель найдена по пути {MODEL_PATH}")

    return MODEL_PATH


def run_inference(**kwargs):
    """Выполнение инференса"""
    # Получение путей к данным и модели
    ti = kwargs['ti']
    data_path = ti.xcom_pull(task_ids='load_inference_data')
    model_path = ti.xcom_pull(task_ids='load_model')

    # Загрузка данных
    df = pd.read_csv(data_path)

    # Загрузка модели
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    # Выполнение предсказаний
    predictions = model.predict(df)

    # Сохранение предсказаний
    result_df = df.copy()
    result_df['prediction'] = predictions

    # Создание директории, если она не существует
    os.makedirs(os.path.dirname(PREDICTIONS_PATH), exist_ok=True)

    # Сохранение результатов
    result_df.to_csv(PREDICTIONS_PATH, index=False)
    print(f"Результаты инференса сохранены в {PREDICTIONS_PATH}")

    return PREDICTIONS_PATH


def save_results(**kwargs):
    """Сохранение результатов инференса"""
    # Получение пути к результатам
    ti = kwargs['ti']
    predictions_path = ti.xcom_pull(task_ids='run_inference')

    # Загрузка результатов
    df = pd.read_csv(predictions_path)

    # Анализ результатов (пример)
    n_samples = len(df)
    predictions_count = df['prediction'].value_counts().to_dict()

    # Сохранение сводки результатов
    summary_path = PREDICTIONS_PATH.replace('.csv', '_summary.json')
    summary = {
        'n_samples': n_samples,
        'predictions_count': predictions_count,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    with open(summary_path, 'w') as f:
        json.dump(summary, f)

    print(f"Сводка результатов сохранена в {summary_path}")
    return summary_path


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
    'inference_pipeline',
    default_args=default_args,
    description='Пайплайн инференса модели машинного обучения',
    schedule_interval=timedelta(days=1),
    catchup=False
)

# Определение задач
load_data_task = PythonOperator(
    task_id='load_inference_data',
    python_callable=load_inference_data,
    provide_context=True,
    dag=dag,
)

load_model_task = PythonOperator(
    task_id='load_model',
    python_callable=load_model,
    provide_context=True,
    dag=dag,
)

inference_task = PythonOperator(
    task_id='run_inference',
    python_callable=run_inference,
    provide_context=True,
    dag=dag,
)

save_results_task = PythonOperator(
    task_id='save_results',
    python_callable=save_results,
    provide_context=True,
    dag=dag,
)

# Определение зависимостей задач
load_data_task >> load_model_task >> inference_task >> save_results_task