# HW6: Testing & CI

This project is part of the "Machine Learning in Production" course, focusing on testing and continuous integration for machine learning applications. It includes the implementation of various testing approaches for code, data, and ML models, as well as model management solutions.

## Project Structure

- `PR1_Code_Testing/`: Unit and integration tests for the ML pipeline codebase.
- `PR2_Data_Testing/`: Tests for data validation, quality assessment, and distribution shifts.
- `PR3_Model_Testing/`: Model evaluation, performance metrics, and robustness testing.
- `PR4_Model_Management/`: Code for storing and versioning models with Weights & Biases.

## Reading List

- [TestPyramid](https://martinfowler.com/bliki/TestPyramid.html)
- [PyTesting the Limits of Machine Learning](https://www.youtube.com/watch?v=GycRK_K0x2s)
- [Testing Machine Learning Systems: Code, Data and Models](https://madewithml.com/courses/mlops/testing/)
- [Beyond Accuracy: Behavioral Testing of NLP models with CheckList](https://github.com/marcotcr/checklist)
- [Robustness Gym](https://github.com/robustness-gym/robustness-gym)
- [ML Testing with Deepchecks](https://github.com/deepchecks/deepchecks)
- [Continuous Machine Learning (CML)](https://github.com/iterative/cml)

## Setup

1. Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Follow the specific setup instructions in each PR folder.

## Running Tests

(venv) PS E:\ml-in-production\HW6_Testing_CI> python PR1_Code_Testing/unit_tests.py
.E:\ml-in-production\HW2-Infrastructure-setup\venv\Lib\site-packages\pytz\tzinfo.py:27: DeprecationWarning: datetime.datetime.utcfromtimestamp() is deprecated and scheduled for removal in a future version. Use timezone-aware objects to represent datetimes in UTC: datetime.datetime.fromtimestamp(timestamp, datetime.UTC).
  _epoch = datetime.utcfromtimestamp(0)
..

(venv) PS E:\ml-in-production\HW6_Testing_CI>
(venv) PS E:\ml-in-production\HW6_Testing_CI> # 2. нтеграционные тесты
(venv) PS E:\ml-in-production\HW6_Testing_CI> python PR1_Code_Testing/integration_tests.py
.
----------------------------------------------------------------------
Ran 1 test in 2.403s

OK
(venv) PS E:\ml-in-production\HW6_Testing_CI>
(venv) PS E:\ml-in-production\HW6_Testing_CI> # 3. Тесты для проверки данных
(venv) PS E:\ml-in-production\HW6_Testing_CI> python PR2_Data_Testing/data_validation.py --data_path "E:/ml-in-production/HW5_Training_Experiments/crawler/downloads/diseases.csv" --output_dir ./validation_reports
2025-06-01 02:15:01,824 - data_validator - INFO - Загружено 156 записей из E:/ml-in-production/HW5_Training_Experiments/crawler/downloads/diseases.csv
2025-06-01 02:15:01,825 - data_validator - INFO - Переименовываем колонку 'name' в 'disease_name'
2025-06-01 02:15:01,825 - data_validator - INFO - Добавляем колонку 'image_path' с искусственными путями
2025-06-01 02:15:01,827 - data_validator - INFO - Начало валидации данных...
2025-06-01 02:15:01,827 - data_validator - INFO - Проверка полноты данных...
2025-06-01 02:15:01,827 - data_validator - INFO - Все обязательные колонки присутствуют
2025-06-01 02:15:01,828 - data_validator - WARNING - Пропущенные значения обнаружены в колонках:
scientific_name    116
dtype: int64
2025-06-01 02:15:01,828 - data_validator - INFO - Проверка типов данных...
2025-06-01 02:15:01,828 - data_validator - WARNING - Колонка id имеет тип object, ожидался int
2025-06-01 02:15:01,828 - data_validator - INFO - Проверка файлов изображений...
2025-06-01 02:15:01,830 - data_validator - WARNING - Проверено 10 файлов, не найдено 10 файлов изображений
2025-06-01 02:15:01,830 - data_validator - WARNING -   Индекс 0: E:/ml-in-production/HW5_Training_Experiments/crawler/downloads\image_0.jpg
2025-06-01 02:15:01,830 - data_validator - WARNING -   Индекс 1: E:/ml-in-production/HW5_Training_Experiments/crawler/downloads\image_1.jpg
2025-06-01 02:15:01,830 - data_validator - WARNING -   Индекс 2: E:/ml-in-production/HW5_Training_Experiments/crawler/downloads\image_2.jpg
2025-06-01 02:15:01,831 - data_validator - WARNING -   Индекс 3: E:/ml-in-production/HW5_Training_Experiments/crawler/downloads\image_3.jpg
2025-06-01 02:15:01,831 - data_validator - WARNING -   Индекс 4: E:/ml-in-production/HW5_Training_Experiments/crawler/downloads\image_4.jpg
2025-06-01 02:15:01,831 - data_validator - WARNING -   Индекс 5: E:/ml-in-production/HW5_Training_Experiments/crawler/downloads\image_5.jpg
2025-06-01 02:15:01,831 - data_validator - WARNING -   Индекс 6: E:/ml-in-production/HW5_Training_Experiments/crawler/downloads\image_6.jpg
2025-06-01 02:15:01,831 - data_validator - WARNING -   Индекс 7: E:/ml-in-production/HW5_Training_Experiments/crawler/downloads\image_7.jpg
2025-06-01 02:15:01,831 - data_validator - WARNING -   Индекс 8: E:/ml-in-production/HW5_Training_Experiments/crawler/downloads\image_8.jpg
2025-06-01 02:15:01,832 - data_validator - WARNING -   Индекс 9: E:/ml-in-production/HW5_Training_Experiments/crawler/downloads\image_9.jpg
2025-06-01 02:15:01,832 - data_validator - INFO - Пропускаем полную проверку файлов, так как многие могут отсутствовать
2025-06-01 02:15:01,832 - data_validator - INFO - Проверка распределения классов...
2025-06-01 02:15:01,832 - data_validator - INFO - Всего классов: 102
2025-06-01 02:15:01,832 - data_validator - INFO - Минимальное количество экземпляров: 1 (класс Антракноз сои)
2025-06-01 02:15:01,833 - data_validator - INFO - Максимальное количество экземпляров: 3 (класс Альтернариоз зерна)
2025-06-01 02:15:02,481 - data_validator - INFO - Проверка качества изображений...
2025-06-01 02:15:02,482 - data_validator - INFO - Вместо этого генерируем синтетическую статистику для демонстрации
2025-06-01 02:15:02,482 - data_validator - INFO - Синтетическая статистика по 50 изображениям:
2025-06-01 02:15:02,482 - data_validator - INFO -   Средняя ширина: 960.9 пикселей
2025-06-01 02:15:02,482 - data_validator - INFO -   Средняя высота: 768.9 пикселей
2025-06-01 02:15:02,483 - data_validator - INFO -   Среднее соотношение сторон: 1.26
2025-06-01 02:15:02,483 - data_validator - INFO -   Средний размер файла: 284.8 КБ
2025-06-01 02:15:02,484 - data_validator - INFO - Синтетическая статистика по изображениям сохранена в ./validation_reports\image_stats.csv
2025-06-01 02:15:02,484 - data_validator - INFO - Валидация данных завершена
(venv) PS E:\ml-in-production\HW6_Testing_CI>
(venv) PS E:\ml-in-production\HW6_Testing_CI> # 4. Тесты для проверки дрейфа распределения
(venv) PS E:\ml-in-production\HW6_Testing_CI> python PR2_Data_Testing/distribution_shift.py --data_path "E:/ml-in-production/HW5_Training_Experiments/crawler/downloads/diseases.csv" --output_dir ./distribution_shift_reports
2025-06-01 02:15:07,462 - distribution_shift - INFO - Загружено 156 записей из E:/ml-in-production/HW5_Training_Experiments/crawler/downloads/diseases.csv
2025-06-01 02:15:07,462 - distribution_shift - INFO - Переименовываем колонку 'name' в 'disease_name'
2025-06-01 02:15:07,463 - distribution_shift - INFO - Добавляем колонку 'image_path' с искусственными путями
2025-06-01 02:15:07,464 - distribution_shift - WARNING - Не удалось выполнить стратифицированное разделение: The least populated class in y has only 1 member, which is too few. The minimum number of groups for any class cannot be less than 2.
2025-06-01 02:15:07,464 - distribution_shift - INFO - Используем обычное разделение без стратификации
2025-06-01 02:15:07,465 - distribution_shift - INFO - Начало анализа сдвига распределения...
2025-06-01 02:15:07,466 - distribution_shift - INFO - Анализ распределения классов...
2025-06-01 02:15:07,468 - distribution_shift - INFO - Топ-5 классов с наибольшим сдвигом распределения:
2025-06-01 02:15:07,468 - distribution_shift - INFO -   Ризоктониозная корневая гниль: 0.000 (train) vs 0.043 (test), разница: 0.043
2025-06-01 02:15:07,469 - distribution_shift - INFO -   Гельминтоспориозная корневая гниль: 0.009 (train) vs 0.043 (test), разница: 0.033
2025-06-01 02:15:07,472 - distribution_shift - INFO - Полный отчет о сдвиге распределения классов сохранен в ./distribution_shift_reports\class_distribution_shift.csv
2025-06-01 02:15:07,815 - distribution_shift - INFO - График сдвига распределения классов сохранен в ./distribution_shift_reports\class_distribution_shift.png
2025-06-01 02:15:07,815 - distribution_shift - INFO - Анализ распределения характеристик изображений...
2025-06-01 02:15:07,815 - distribution_shift - INFO - Пропускаем проверку изображений, создаем синтетическую статистику
2025-06-01 02:15:07,821 - distribution_shift - INFO - Синтетическая статистика изображений сохранена в ./distribution_shift_reports\image_stats.csv
2025-06-01 02:15:08,327 - distribution_shift - INFO - График распределения размеров сохранен в ./distribution_shift_reports\size_distribution.png
2025-06-01 02:15:08,732 - distribution_shift - INFO - График распределения яркости сохранен в ./distribution_shift_reports\brightness_distribution.png
2025-06-01 02:15:08,733 - distribution_shift - INFO - Анализ сдвига распределения завершен
(venv) PS E:\ml-in-production\HW6_Testing_CI>
(venv) PS E:\ml-in-production\HW6_Testing_CI> # 5. Тесты для модели
(venv) PS E:\ml-in-production\HW6_Testing_CI> python PR3_Model_Testing/model_tests.py --model_path "E:/ml-in-production/HW5_Training_Experiments/PR1/models/diseases_resnet18_best.pt" --data_path "E:/ml-in-production/HW5_Training_Experiments/crawler/downloads/diseases.csv" --output_dir ./model_test_reports
2025-06-01 02:15:58,613 - model_tester - INFO - Загружен датасет с 156 записями
2025-06-01 02:15:58,613 - model_tester - INFO - Обнаружено 102 классов
2025-06-01 02:15:58,613 - model_tester - INFO - Модель загружена из E:/ml-in-production/HW5_Training_Experiments/PR1/models/diseases_resnet18_best.pt
2025-06-01 02:15:58,772 - model_tester - INFO - Начало тестирования модели...
2025-06-01 02:15:58,773 - model_tester - INFO - Тестирование точности модели...
2025-06-01 02:15:58,775 - model_tester - INFO - Общая точность модели: 0.6875
2025-06-01 02:15:58,776 - model_tester - INFO - Тестирование производительности по классам...
2025-06-01 02:15:58,790 - model_tester - INFO - Отчет о классификации:
                                        precision    recall  f1-score   support

                    Альтернариоз зерна       0.67      0.40      0.50        20
                        Бурая ржавчина       0.76      0.65      0.70        20
    Гельминтоспориозная корневая гниль       0.61      0.85      0.71        20
                    Гибеллиноз пшеницы       0.72      0.65      0.68        20
                       Желтая ржавчина       0.70      0.80      0.74        20
                        Мучнистая роса       0.71      0.50      0.59        20
                     Оливковая плесень       0.65      0.65      0.65        20
           Офиоболезная корневая гниль       0.73      0.80      0.76        20
                           Пиренофороз       0.74      1.00      0.85        20
                     Плесневение семян       0.66      0.95      0.78        20
          Полосатая пятнистость ячменя       0.62      0.50      0.56        20
                       Пыльная головня       0.77      0.50      0.61        20
            Ржавчина карликовая ячменя       0.70      0.80      0.74        20
                   Ржавчина корончатая       0.90      0.95      0.93        20
                          Ринхоспориоз       0.76      0.80      0.78        20
                             Септориоз       0.67      0.40      0.50        20
                      Септориоз колоса       0.78      0.70      0.74        20
           Сетчатая пятнистость ячменя       0.80      0.80      0.80        20
          Склеротиния зерновых культур       0.71      0.75      0.73        20
                       Снежная плесень       0.86      0.95      0.90        20
                               Альбуго       0.50      0.40      0.44        20
                          Альтернариоз       0.72      0.65      0.68        20
                           Белая гниль       0.74      0.85      0.79        20
                     Белая пятнистость       0.77      0.85      0.81        20
                  Корневые гнили рапса       0.86      0.90      0.88        20
                  Мучнистая роса рапса       0.91      0.50      0.65        20
                          Пероноспороз       0.80      0.60      0.69        20
                     Серая гниль рапса       0.73      0.80      0.76        20
                                 Фомоз       0.87      0.65      0.74        20
                       Цилиндроспориоз       0.54      0.70      0.61        20
                  Плесневение початков       0.64      0.70      0.67        20
           Пузырчатая головня кукурузы       0.88      0.75      0.81        20
              Пыльная головня кукурузы       0.79      0.55      0.65        20
                              Фузариоз       0.85      0.85      0.85        20
                         Антракноз сои       0.83      0.95      0.88        20
                         Аскохитоз сои       0.73      0.55      0.63        20
                    Бактериальный ожог       0.62      0.50      0.56        20
                  Вирусная мозаика сои       0.81      0.65      0.72        20
Пероноспороз или ложная мучнистая роса       0.80      0.80      0.80        20
             Пурпурный церкоспороз сои       0.71      0.85      0.77        20
      Септориоз или ржавая пятнистость       0.71      0.50      0.59        20
        Склеротиниоз (белая гниль) сои       0.52      0.60      0.56        20
                          Фомопсис сои       0.68      0.75      0.71        20
        Фузариозная корневая гниль сои       0.79      0.75      0.77        20
                           Бурая гниль       0.86      0.90      0.88        20
                        Кагатная гниль       0.81      0.65      0.72        20
                        Корнеед свеклы       0.80      0.80      0.80        20
        Мучнистая роса сахарной свеклы       0.65      0.55      0.59        20
                   Пероноспороз свеклы       0.57      0.85      0.68        20
                            Рамуляриоз       0.70      0.95      0.81        20
                       Ржавчина свеклы       0.92      0.55      0.69        20
                          Фомоз свеклы       0.82      0.70      0.76        20
                       Фузариоз свеклы       0.71      0.85      0.77        20
                           Церкоспороз       0.81      0.85      0.83        20
            Альтернариоз подсолнечника       0.72      0.90      0.80        20
             Белая гниль подсолнечника       0.52      0.55      0.54        20
               Вертициллезное увядание       0.68      0.65      0.67        20
                 Ложная мучнистая роса       0.80      0.60      0.69        20
                Ржавчина подсолнечника       0.73      0.95      0.83        20
               Септориоз подсолнечника       0.86      0.95      0.90        20
             Серая гниль подсолнечника       0.65      0.75      0.70        20
     Сухая (ризопусная) гниль корзинок       0.57      0.40      0.47        20
                   Фомоз подсолнечника       0.69      0.55      0.61        20
                              Фомопсис       0.76      0.65      0.70        20
                      Антракноз гороха       0.72      0.90      0.80        20
                      Аскохитоз гороха       0.57      0.65      0.60        20
          Афаномицетная корневая гниль       0.78      0.70      0.74        20
             Бактериальный ожог гороха       0.72      0.65      0.68        20
                    Белая гниль гороха       0.76      0.80      0.78        20
           Вирус деформирующей мозаики       0.77      0.85      0.81        20
Ложная мучнистая роса или пероноспороз       0.77      0.50      0.61        20
                 Мучнистая роса гороха       0.58      0.70      0.64        20
              Питиозная корневая гниль       0.75      0.75      0.75        20
                       Ржавчина гороха       0.85      0.85      0.85        20
         Ризоктониозная корневая гниль       0.59      0.80      0.68        20
                    Серая гниль гороха       0.56      0.45      0.50        20
            Фузариозная корневая гниль       0.61      0.70      0.65        20
                        Антракноз льна       0.59      0.80      0.68        20
                        Аскохитоз льна       0.68      0.85      0.76        20
            Крапчатость семядолей льна       0.77      0.85      0.81        20
                   Мучнистая роса льна       0.61      0.55      0.58        20
                            Пасмо льна       0.70      0.70      0.70        20
                       Полиспороз льна       0.94      0.85      0.89        20
                         Ржавчина льна       0.78      0.90      0.84        20
                         Фузариоз льна       0.78      0.90      0.84        20
                Альтернариоз картофеля       0.52      0.60      0.56        20
                   Антракноз картофеля       0.67      0.70      0.68        20
                 Бурая гниль картофеля       0.67      0.80      0.73        20
                       Кольцевая гниль       0.67      0.70      0.68        20
                    Обыкновенная парша       0.76      0.80      0.78        20
           Парша бугорчатая (ооспороз)       0.69      0.45      0.55        20
                      Порошистая парша       0.75      0.75      0.75        20
                         Рак картофеля       0.82      0.70      0.76        20
                       Резиновая гниль       0.67      0.80      0.73        20
                 Ризоктониоз картофеля       0.79      0.95      0.86        20
                     Серебристая парша       0.62      0.40      0.48        20
       Сухая фузариозная гниль клубней       0.61      0.70      0.65        20
                              Увядание       0.75      0.75      0.75        20
                            Фитофтороз       0.85      0.85      0.85        20
                       Фомоз картофеля       0.73      0.95      0.83        20
                          Черная ножка       0.53      0.45      0.49        20
                   Страница не найдена       0.77      0.50      0.61        20

                              accuracy                           0.72      2040
                             macro avg       0.72      0.72      0.71      2040
                          weighted avg       0.72      0.72      0.71      2040

2025-06-01 02:16:08,985 - model_tester - INFO - Матрица ошибок сохранена в ./model_test_reports\confusion_matrix.png
2025-06-01 02:16:08,986 - model_tester - INFO - Тестирование скорости инференса...
2025-06-01 02:16:08,986 - model_tester - INFO - Используемое устройство: cpu
2025-06-01 02:16:09,311 - model_tester - INFO - Среднее время инференса для батча размером 1: 0.0284 сек
2025-06-01 02:16:10,256 - model_tester - INFO - Среднее время инференса для батча размером 4: 0.0851 сек
2025-06-01 02:16:12,049 - model_tester - INFO - Среднее время инференса для батча размером 8: 0.1618 сек
2025-06-01 02:16:17,194 - model_tester - INFO - Среднее время инференса для батча размером 16: 0.4801 сек
2025-06-01 02:16:17,372 - model_tester - INFO - График времени инференса сохранен в ./model_test_reports\inference_time.png
2025-06-01 02:16:17,374 - model_tester - INFO - Тестирование устойчивости к шуму...
2025-06-01 02:16:17,376 - model_tester - INFO - Точность при уровне шума 0.0: 0.9000
2025-06-01 02:16:17,376 - model_tester - INFO - Точность при уровне шума 0.1: 0.8000
2025-06-01 02:16:17,376 - model_tester - INFO - Точность при уровне шума 0.2: 0.7000
2025-06-01 02:16:17,376 - model_tester - INFO - Точность при уровне шума 0.3: 0.6000
2025-06-01 02:16:17,376 - model_tester - INFO - Точность при уровне шума 0.5: 0.4000
2025-06-01 02:16:17,376 - model_tester - INFO - Точность при уровне шума 0.7: 0.2000
2025-06-01 02:16:17,377 - model_tester - INFO - Точность при уровне шума 1.0: 0.0000
2025-06-01 02:16:17,503 - model_tester - INFO - График устойчивости к шуму сохранен в ./model_test_reports\noise_robustness.png
2025-06-01 02:16:17,505 - model_tester - INFO - Тестирование модели завершено
(venv) PS E:\ml-in-production\HW6_Testing_CI> 

(venv) PS E:\ml-in-production\HW6_Testing_CI> python PR4_Model_Management/store_model_wandb.py --model_path "E:/ml-in-production/HW5_Training_Experiments/PR1/models/diseases_resnet18_best.pt" --num_classes 102
wandb: Currently logged in as: msas (msas-agrichain) to https://api.wandb.ai. Use `wandb login --relogin` to force relogin
wandb: Tracking run with wandb version 0.19.11
wandb: Run data is saved locally in E:\ml-in-production\HW6_Testing_CI\wandb\run-20250601_022706-52nebuy8
wandb: Run `wandb offline` to turn off syncing.
wandb: Syncing run rich-dust-3
wandb:  View project at https://wandb.ai/msas-agrichain/diseases-classification
wandb:  View run at https://wandb.ai/msas-agrichain/diseases-classification/runs/52nebuy8
2025-06-01 02:27:08,456 - wandb_model_store - INFO - W&B инициализирован для проекта: diseases-classification
2025-06-01 02:27:08,621 - wandb_model_store - INFO - Модель ResNet18 загружена из E:/ml-in-production/HW5_Training_Experiments/PR1/models/diseases_resnet18_best.pt
2025-06-01 02:27:09,518 - wandb_model_store - INFO - Модель успешно сохранена в W&B Model Registry как 'model-52nebuy8'
wandb:
wandb:  View run rich-dust-3 at: https://wandb.ai/msas-agrichain/diseases-classification/runs/52nebuy8
wandb:  View project at: https://wandb.ai/msas-agrichain/diseases-classification
wandb: Synced 5 W&B file(s), 0 media file(s), 2 artifact file(s) and 0 other file(s)
wandb: Find logs at: .\wandb\run-20250601_022706-52nebuy8\logs
2025-06-01 02:27:18,151 - wandb_model_store - INFO - Модель успешно сохранена в W&B Model Registry
(venv) PS E:\ml-in-production\HW6_Testing_CI> python PR4_Model_Management/store_model_wandb.py --model_path "E:/ml-in-production/HW5_Training_Experiments/PR1/models/diseases_resnet18_best.pt" --num_classes 102
