# HW8 - Dagster ML Пайплайни

Цей проєкт демонструє використання Dagster для створення пайплайнів машинного навчання. 
Проєкт містить два основні пайплайни та набір активів (assets):
1. Пайплайн навчання моделі
2. Пайплайн інференсу
3. Активи для даних та моделей

## Вимоги

- Python 3.12.8
- Віртуальне середовище (virtualenv)
- Залежності з requirements.txt

## Встановлення

1. Клонуйте репозиторій
   ```bash
   git clone <repository-url>
   cd HW8_Dagster
   ```

2. Активуйте віртуальне середовище
   ```bash
   # На Windows
   .\venv\Scripts\Activate.ps1
   
   # На Linux/Mac
   source venv/bin/activate
   ```

3. Встановіть залежності
   ```bash
   pip install -r requirements.txt
   ```

## Команди для запуску пайплайнів

### Запуск Dagster веб-інтерфейсу

``` bash
dagster dev
``` 

### Запуск пайплайну навчання моделі
``` bash
# Через CLI
dagster job execute -f PR1_Dagster_Training_Pipeline/training_pipeline.py -a training_pipeline

# З параметрами
dagster job execute -f PR1_Dagster_Training_Pipeline/training_pipeline.py -a training_pipeline --config PR1_Dagster_Training_Pipeline/training_config.yaml
```


### Запуск пайплайну навчання нейронної мережі
``` bash
dagster job execute -f PR1_Dagster_Training_Pipeline/training_pipeline.py -a training_pipeline
```
### Запуск пайплайну інференсу
``` bash
dagster job execute -f PR2_Dagster_Inference_Pipeline/inference_pipeline.py -a inference_pipeline
```




### Матеріалізація активів (assets)
``` bash
# Матеріалізація активу crawled_dataset
dagster asset materialize --select crawled_dataset

# Матеріалізація активу trained_model
dagster asset materialize --select trained_model

# Матеріалізація всіх активів
dagster asset materialize --all


```

