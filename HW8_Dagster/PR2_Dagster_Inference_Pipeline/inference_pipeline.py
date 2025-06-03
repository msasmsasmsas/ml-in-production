# inference_pipeline.py
# inference_pipeline.py

import os
import pandas as pd
import pickle
import numpy as np
from typing import Any
from dagster import job, op, In, Out, Output, get_dagster_logger, Field


@op(
    out=Out(),
    config_schema={
        "data_path": Field(str, default_value="data/inference_data.csv",
                           description="Шлях до файлу з даними для інференсу")
    }
)
def load_inference_data(context):
    logger = get_dagster_logger()
    config = context.op_config

    logger.info(f"Завантаження даних для інференсу з {config['data_path']}")

    try:
        if config["data_path"].endswith('.csv'):
            df = pd.read_csv(config["data_path"])
        elif config["data_path"].endswith('.parquet'):
            df = pd.read_parquet(config["data_path"])
        else:
            raise ValueError(f"Непідтримуваний формат файлу: {config['data_path']}")
    except Exception as e:
        logger.warning(f"Помилка при завантаженні даних: {e}. Використовуємо синтетичні дані.")
        # Створюємо синтетичні дані для демонстрації
        X = np.random.rand(100, 10)
        df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])

    logger.info(f"Завантажено {len(df)} записів для інференсу")
    return df


@op(
    out=Out(),
    config_schema={
        "model_path": Field(str, default_value="models/model.pkl",
                           description="Шлях до збереженої моделі")
    }
)
def load_model(context):
    logger = get_dagster_logger()
    config = context.op_config

    model_path = config["model_path"]
    logger.info(f"Завантаження моделі з {model_path}")

    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        logger.info(f"Модель успішно завантажено")
        return model
    except Exception as e:
        logger.error(f"Помилка при завантаженні моделі: {e}")
        raise


@op(
    ins={"data": In(), "model": In()},
    out=Out(),
    config_schema={
        "batch_size": Field(int, default_value=32,
                           description="Розмір батчу для інференсу")
    }
)
def run_inference(context, data, model):
    logger = get_dagster_logger()
    config = context.op_config

    logger.info(f"Запуск інференсу на {len(data)} прикладах")
    
    # Запускаємо інференс
    try:
        # Перевіряємо тип моделі (sklearn або pytorch)
        if hasattr(model, 'predict'):
            # Для sklearn моделей
            predictions = model.predict(data.values)
            
            # Якщо є метод predict_proba, отримаємо ймовірності
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(data.values)
                
                # Створюємо DataFrame з результатами
                results_df = pd.DataFrame(data)
                results_df['prediction'] = predictions
                
                # Додаємо ймовірності
                if probabilities.shape[1] == 2:  # Бінарна класифікація
                    results_df['probability'] = probabilities[:, 1]
                else:  # Мультикласова класифікація
                    for i in range(probabilities.shape[1]):
                        results_df[f'probability_class_{i}'] = probabilities[:, i]
            else:
                # Без ймовірностей
                results_df = pd.DataFrame(data)
                results_df['prediction'] = predictions
        else:
            # Для PyTorch моделей
            import torch
            
            # Переводимо модель в режим оцінки
            model.eval()
            
            # Створюємо датасет та DataLoader
            batch_size = config["batch_size"]
            predictions = []
            
            with torch.no_grad():
                for i in range(0, len(data), batch_size):
                    batch = data.iloc[i:i+batch_size]
                    inputs = torch.tensor(batch.values, dtype=torch.float32)
                    outputs = model(inputs)
                    preds = (outputs > 0.5).float().numpy()
                    predictions.extend(preds)
                    
            results_df = pd.DataFrame(data)
            results_df['prediction'] = predictions
    
    except Exception as e:
        logger.error(f"Помилка при виконанні інференсу: {e}")
        raise
    
    logger.info(f"Інференс завершено, отримано {len(results_df)} результатів")
    return results_df


@op(
    ins={"results": In()},
    out=Out(str),
    config_schema={
        "output_path": Field(str, default_value="results/inference_results.csv",
                            description="Шлях для збереження результатів"),
        "save_metadata": Field(bool, default_value=True,
                              description="Чи зберігати метадані")
    }
)
def save_inference_results(context, results):
    logger = get_dagster_logger()
    config = context.op_config

    # Створюємо директорію для результатів
    output_dir = os.path.dirname(config["output_path"])
    os.makedirs(output_dir, exist_ok=True)
    
    # Зберігаємо результати
    results.to_csv(config["output_path"], index=False)
    logger.info(f"Результати інференсу збережено у {config['output_path']}")
    
    # Зберігаємо метадані, якщо потрібно
    if config["save_metadata"]:
        import json
        from datetime import datetime
        
        metadata = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "record_count": len(results),
            "columns": list(results.columns),
            "predictions_summary": {
                "count": int(results["prediction"].count()),
                "mean": float(results["prediction"].mean()) if "prediction" in results else None,
                "positive_ratio": float((results["prediction"] > 0.5).mean()) if "prediction" in results else None
            }
        }
        
        metadata_path = os.path.splitext(config["output_path"])[0] + "_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        
        logger.info(f"Метадані інференсу збережено у {metadata_path}")
    
    return config["output_path"]


@job
def inference_pipeline():
    data = load_inference_data()
    model = load_model()
    results = run_inference(data, model)
    save_inference_results(results)
from dagster import job, op, In, Out, get_dagster_logger, Field


@op(
    out=Out(list),
    config_schema={
        "data_source": Field(str, default_value="test_data",
                             description="Джерело даних для інференсу"),
        "num_samples": Field(int, default_value=10,
                             description="Кількість зразків для інференсу")
    }
)
def load_data_for_inference(context):
    logger = get_dagster_logger()
    config = context.op_config
    logger.info(f"Завантаження даних для інференсу з {config['data_source']}...")

    # Штучні дані для передбачення
    inference_data = list(range(config["num_samples"]))
    logger.info(f"Завантажено {len(inference_data)} зразків для інференсу")

    return inference_data


@op(
    out=Out(),
    config_schema={
        "model_path": Field(str, default_value="models/model.pkl",
                            description="Шлях до збереженої моделі")
    }
)
def load_trained_model(context):
    logger = get_dagster_logger()
    config = context.op_config
    logger.info(f"Завантаження навченої моделі з {config['model_path']}...")

    try:
        import pickle
        with open(config["model_path"], 'rb') as f:
            model = pickle.load(f)
        logger.info("Модель успішно завантажена")
    except Exception as e:
        logger.warning(f"Помилка при завантаженні моделі: {e}. Використовуємо імітацію моделі.")
        # Імітація завантаження навченої моделі
        model = {"coeff": 2}

    return model



@op(
    ins={
        "data": In(dagster_type=list),
        "model": In(dagster_type=Any)  # Изменить с dict на Any
    },
    out=Out(),
)
def run_inference(context, data, model):
    logger = get_dagster_logger()
    logger.info(f"Запуск інференсу з моделлю {model}...")

    # Перевіряємо тип моделі та виконуємо відповідний інференс
    if hasattr(model, 'predict'):
        # Якщо це scikit-learn модель
        try:
            import numpy as np
            results = model.predict(np.array(data).reshape(-1, 1))
            results = results.tolist()
        except:
            logger.warning("Помилка під час інференсу scikit-learn моделі. Використовуємо просту модель.")
            results = [2 * x for x in data]  # Проста модель: y = 2 * x
    elif isinstance(model, dict) and 'coeff' in model:
        # Проста модель: y = coeff * x
        results = [model['coeff'] * x for x in data]
    else:
        logger.warning("Невідомий тип моделі. Використовуємо стандартний інференс.")
        results = [x + 1 for x in data]  # Проста трансформація

    logger.info(f"Інференс завершено. Результати: {results}")
    return results


@op(
    ins={'results': In(list)},
    out=Out(bool),
    config_schema={
        "output_path": Field(str, default_value="results/inference_results.txt",
                             description="Шлях для збереження результатів інференсу")
    }
)
def save_inference_results(context, results):
    logger = get_dagster_logger()
    config = context.op_config
    logger.info(f"Збереження результатів інференсу: {results}")

    try:
        # Створюємо директорію, якщо вона не існує
        import os
        os.makedirs(os.path.dirname(config["output_path"]), exist_ok=True)

        # Зберігаємо результати
        with open(config["output_path"], 'w') as f:
            f.write(f"Результати інференсу:\n")
            for i, result in enumerate(results):
                f.write(f"Зразок {i}: {result}\n")

        logger.info(f"Результати успішно збережено у {config['output_path']}")
        return True
    except Exception as e:
        logger.error(f"Помилка при збереженні результатів: {e}")
        return False


@job
def inference_pipeline():
    data = load_data_for_inference()
    model = load_trained_model()
    results = run_inference(data, model)
    save_inference_results(results)