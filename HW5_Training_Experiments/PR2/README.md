# PR2: Пошук гіперпараметрів з W&B Sweeps

Цей модуль реалізує процес пошуку оптимальних гіперпараметрів для моделі класифікації сільськогосподарських ризиків з використанням Weights & Biases Sweeps.

## Функціональність

- Автоматичний пошук оптимальних гіперпараметрів моделі
- Підтримка різних стратегій пошуку (випадковий, решітчастий, байєсівський)
- Логування результатів експериментів у W&B
- Візуалізація впливу гіперпараметрів на продуктивність моделі

## Структура

- `sweep.py` - основний скрипт для запуску пошуку гіперпараметрів
- `models/` - директорія для збереження навчених моделей
- `README.md` - документація

## Використання

1. Встановіть необхідні залежності:


💻 Пристрій: cpu
🔄 Створення завантажувачів даних...
🔄 Завантаження даних: diseases
📊 Ризиків: 156, Зображень: 152
✅ Після фільтрації: 86 зображень, 33 класів
📊 Розділення: 60 тренувальних, 26 валідаційних
✅ Готово! Класів: 33, Батчів: train=2, val=1
🧠 Створення моделі resnet18 для 33 класів
🧊 Backbone заморожено
🚀 Початок навчання на 15 епох
Епоха 1: Train Acc=0.000, Val F1=0.015                                                                                                                                                      
Епоха 2: Train Acc=0.200, Val F1=0.038                                                                                                                                                      
Епоха 3: Train Acc=0.317, Val F1=0.077                                                                                                                                                      
🔥 Модель розморожена
Епоха 4: Train Acc=0.267, Val F1=0.269                                                                                                                                                      
Епоха 5: Train Acc=0.867, Val F1=0.215                                                                                                                                                      
Епоха 6: Train Acc=0.983, Val F1=0.513                                                                                                                                                      
Епоха 7: Train Acc=0.967, Val F1=0.417                                                                                                                                                      
Епоха 8: Train Acc=0.950, Val F1=0.654                                                                                                                                                      
Епоха 9: Train Acc=0.967, Val F1=0.438                                                                                                                                                      
Епоха 10: Train Acc=0.950, Val F1=0.494                                                                                                                                                     
Епоха 11: Train Acc=1.000, Val F1=0.419                                                                                                                                                     
Епоха 12: Train Acc=0.983, Val F1=0.296                                                                                                                                                     
Епоха 13: Train Acc=1.000, Val F1=0.335                                                                                                                                                     
⏹️ Early stopping після 13 епох
🎉 Найкращий F1: 0.654
wandb:
wandb:                                                                                                                                                                                      
wandb: Run history:                                                                                                                                                                         
wandb:      epoch ▁▂▂▃▃▄▅▅▆▆▇▇█
wandb:   final_f1 ▁
wandb:  train_acc ▁▂▃▃▇████████
wandb: train_loss █▇▆▆▂▁▁▁▁▁▁▁▁
wandb:    val_acc ▁▁▁▃▃▇▆█▆▆▆▄▅
wandb:     val_f1 ▁▁▂▄▃▆▅█▆▆▅▄▄
wandb:   val_loss ▇▇▇▄▄▂▁▁▃▆███
wandb:
wandb: Run summary:
wandb:      epoch 12
wandb:   final_f1 0.65385
wandb:  train_acc 1
wandb: train_loss 0.03903
wandb:    val_acc 0.38462
wandb:     val_f1 0.33462
wandb:   val_loss 3.75105
wandb:
wandb:  View run super-sweep-10 at: https://wandb.ai/msas-agrichain/agri-risk-hyperparameter-search/runs/t3jf5mya
wandb:  View project at: https://wandb.ai/msas-agrichain/agri-risk-hyperparameter-search
wandb: Synced 5 W&B file(s), 0 media file(s), 0 artifact file(s) and 0 other file(s)
wandb: Find logs at: .\wandb\run-20250524_223750-t3jf5mya\logs
wandb: Sweep Agent: Waiting for job.
wandb: Job received.
wandb: Agent Starting Run: 0c7obzvq with config:
wandb:  batch_size: 16
wandb:  dropout: 0.24446433691413944
wandb:  learning_rate: 0.0016747259583273057
wandb:  model_name: resnet34
wandb:  num_epochs: 20
wandb:  validation_split: 0.3
wandb:  weight_decay: 3.336810474135992e-05
wandb: WARNING Ignoring project 'agri-risk-hyperparameter-search' when running a sweep.
wandb: Tracking run with wandb version 0.19.11
wandb: Run data is saved locally in E:\ml-in-production\HW5_Training_Experiments\pr2\wandb\run-20250524_223940-0c7obzvq
wandb: Run `wandb offline` to turn off syncing.
wandb: Syncing run good-sweep-11
wandb:  View project at https://wandb.ai/msas-agrichain/agri-risk-hyperparameter-search
wandb:  View sweep at https://wandb.ai/msas-agrichain/agri-risk-hyperparameter-search/sweeps/m1p4k5l2
wandb:  View run at https://wandb.ai/msas-agrichain/agri-risk-hyperparameter-search/runs/0c7obzvq
💻 Пристрій: cpu
🔄 Створення завантажувачів даних...
🔄 Завантаження даних: diseases
📊 Ризиків: 156, Зображень: 152
✅ Після фільтрації: 86 зображень, 33 класів
📊 Розділення: 60 тренувальних, 26 валідаційних
✅ Готово! Класів: 33, Батчів: train=4, val=2
🧠 Створення моделі resnet34 для 33 класів
🧊 Backbone заморожено
🚀 Початок навчання на 20 епох
Епоха 1: Train Acc=0.033, Val F1=0.000                                                                                                                                                      
Епоха 2: Train Acc=0.150, Val F1=0.096                                                                                                                                                      
Епоха 3: Train Acc=0.300, Val F1=0.160                                                                                                                                                      
🔥 Модель розморожена
Епоха 4: Train Acc=0.400, Val F1=0.000                                                                                                                                                      
Епоха 5: Train Acc=0.533, Val F1=0.031                                                                                                                                                      
Епоха 6: Train Acc=0.450, Val F1=0.003                                                                                                                                                      
Епоха 7: Train Acc=0.583, Val F1=0.000                                                                                                                                                      
Епоха 8: Train Acc=0.500, Val F1=0.077                                                                                                                                                      
⏹️ Early stopping після 8 епох
🎉 Найкращий F1: 0.160
wandb:
wandb:                                                                                                                                                                                      
wandb: Run history:
wandb:      epoch ▁▂▃▄▅▆▇█
wandb:   final_f1 ▁
wandb:  train_acc ▁▂▄▆▇▆█▇
wandb: train_loss █▆▄▃▂▂▁▃
wandb:    val_acc ▁▅█▁▄▂▁▄
wandb:     val_f1 ▁▅█▁▂▁▁▄
wandb:   val_loss ▁▁▁▃██▂▃
wandb:
wandb: Run summary:
wandb:      epoch 7
wandb:   final_f1 0.16026
wandb:  train_acc 0.5
wandb: train_loss 2.12656
wandb:    val_acc 0.07692
wandb:     val_f1 0.07692
wandb:   val_loss 27.01091
wandb:
wandb:  View run good-sweep-11 at: https://wandb.ai/msas-agrichain/agri-risk-hyperparameter-search/runs/0c7obzvq
wandb:  View project at: https://wandb.ai/msas-agrichain/agri-risk-hyperparameter-search
wandb: Synced 5 W&B file(s), 0 media file(s), 0 artifact file(s) and 0 other file(s)
wandb: Find logs at: .\wandb\run-20250524_223940-0c7obzvq\logs
wandb: Agent Starting Run: zy7iwqnm with config:
wandb:  batch_size: 16
wandb:  dropout: 0.3798528118198463
wandb:  learning_rate: 0.0009908996375523746
wandb:  model_name: resnet18
wandb:  num_epochs: 15
wandb:  validation_split: 0.2
wandb:  weight_decay: 0.005409536359022567
wandb: WARNING Ignoring project 'agri-risk-hyperparameter-search' when running a sweep.
wandb: Tracking run with wandb version 0.19.11
wandb: Run data is saved locally in E:\ml-in-production\HW5_Training_Experiments\pr2\wandb\run-20250524_224101-zy7iwqnm
wandb: Run `wandb offline` to turn off syncing.
wandb: Syncing run vocal-sweep-12
wandb:  View project at https://wandb.ai/msas-agrichain/agri-risk-hyperparameter-search
wandb:  View sweep at https://wandb.ai/msas-agrichain/agri-risk-hyperparameter-search/sweeps/m1p4k5l2
wandb:  View run at https://wandb.ai/msas-agrichain/agri-risk-hyperparameter-search/runs/zy7iwqnm
💻 Пристрій: cpu
🔄 Створення завантажувачів даних...
🔄 Завантаження даних: diseases
📊 Ризиків: 156, Зображень: 152
✅ Після фільтрації: 86 зображень, 33 класів
📊 Розділення: 68 тренувальних, 18 валідаційних
✅ Готово! Класів: 33, Батчів: train=5, val=2
🧠 Створення моделі resnet18 для 33 класів
🧊 Backbone заморожено
🚀 Початок навчання на 15 епох
Епоха 1: Train Acc=0.015, Val F1=0.056                                                                                                                                                      
Епоха 2: Train Acc=0.132, Val F1=0.056                                                                                                                                                      
Епоха 3: Train Acc=0.309, Val F1=0.056                                                                                                                                                      
🔥 Модель розморожена
Епоха 4: Train Acc=0.529, Val F1=0.241                                                                                                                                                      
Епоха 5: Train Acc=0.868, Val F1=0.328                                                                                                                                                      
Епоха 6: Train Acc=0.868, Val F1=0.119                                                                                                                                                      
Епоха 7: Train Acc=0.956, Val F1=0.272                                                                                                                                                      
Епоха 8: Train Acc=0.897, Val F1=0.630                                                                                                                                                      
Епоха 9: Train Acc=0.853, Val F1=0.676                                                                                                                                                      
Епоха 10: Train Acc=0.882, Val F1=0.370                                                                                                                                                     
Епоха 11: Train Acc=0.750, Val F1=0.519                                                                                                                                                     
Епоха 12: Train Acc=0.882, Val F1=0.389                                                                                                                                                     
Епоха 13: Train Acc=0.809, Val F1=0.220                                                                                                                                                     
Епоха 14: Train Acc=0.897, Val F1=0.370                                                                                                                                                     
⏹️ Early stopping після 14 епох
🎉 Найкращий F1: 0.676
wandb:
wandb:                                                                                                                                                                                      
wandb: Run history:
wandb:      epoch ▁▂▂▃▃▄▄▅▅▆▆▇▇█
wandb:   final_f1 ▁
wandb:  train_acc ▁▂▃▅▇▇██▇▇▆▇▇█
wandb: train_loss █▇▆▄▁▁▁▁▁▁▂▂▁▁
wandb:    val_acc ▁▁▁▃▅▂▄▇█▅▆▅▄▅
wandb:     val_f1 ▁▁▁▃▄▂▃▇█▅▆▅▃▅
wandb:   val_loss ▃▄▃▂▃█▂▁▁▃▁▂▃▂
wandb:
wandb: Run summary:
wandb:      epoch 13
wandb:   final_f1 0.67593
wandb:  train_acc 0.89706
wandb: train_loss 0.69029
wandb:    val_acc 0.38889
wandb:     val_f1 0.37037
wandb:   val_loss 2.58455
wandb:
wandb:  View run vocal-sweep-12 at: https://wandb.ai/msas-agrichain/agri-risk-hyperparameter-search/runs/zy7iwqnm
wandb:  View project at: https://wandb.ai/msas-agrichain/agri-risk-hyperparameter-search
wandb: Synced 5 W&B file(s), 0 media file(s), 0 artifact file(s) and 0 other file(s)
wandb: Find logs at: .\wandb\run-20250524_224101-zy7iwqnm\logs
wandb: Agent Starting Run: okcp7r9p with config:
wandb:  batch_size: 16
wandb:  dropout: 0.32804455539524313
wandb:  learning_rate: 0.00021209155149515784
wandb:  model_name: resnet34
wandb:  num_epochs: 20
wandb:  validation_split: 0.2
wandb:  weight_decay: 2.055082583330047e-05
wandb: WARNING Ignoring project 'agri-risk-hyperparameter-search' when running a sweep.
wandb: Tracking run with wandb version 0.19.11
wandb: Run data is saved locally in E:\ml-in-production\HW5_Training_Experiments\pr2\wandb\run-20250524_224240-okcp7r9p
wandb: Run `wandb offline` to turn off syncing.
wandb: Syncing run clean-sweep-13
wandb:  View project at https://wandb.ai/msas-agrichain/agri-risk-hyperparameter-search
wandb:  View sweep at https://wandb.ai/msas-agrichain/agri-risk-hyperparameter-search/sweeps/m1p4k5l2
wandb:  View run at https://wandb.ai/msas-agrichain/agri-risk-hyperparameter-search/runs/okcp7r9p
💻 Пристрій: cpu
🔄 Створення завантажувачів даних...
🔄 Завантаження даних: diseases
📊 Ризиків: 156, Зображень: 152
✅ Після фільтрації: 86 зображень, 33 класів
📊 Розділення: 68 тренувальних, 18 валідаційних
✅ Готово! Класів: 33, Батчів: train=5, val=2
🧠 Створення моделі resnet34 для 33 класів
🧊 Backbone заморожено
🚀 Початок навчання на 20 епох
Епоха 1: Train Acc=0.044, Val F1=0.016                                                                                                                                                      
Епоха 2: Train Acc=0.059, Val F1=0.056                                                                                                                                                      
Епоха 3: Train Acc=0.029, Val F1=0.037                                                                                                                                                      
🔥 Модель розморожена
Епоха 4: Train Acc=0.294, Val F1=0.667                                                                                                                                                      
Епоха 5: Train Acc=0.809, Val F1=0.833                                                                                                                                                      
Епоха 6: Train Acc=0.912, Val F1=0.889                                                                                                                                                      
Епоха 7: Train Acc=0.971, Val F1=0.889                                                                                                                                                      
Епоха 8: Train Acc=1.000, Val F1=0.861                                                                                                                                                      
Епоха 9: Train Acc=1.000, Val F1=0.861                                                                                                                                                      
Епоха 10: Train Acc=0.985, Val F1=0.861                                                                                                                                                     
Епоха 11: Train Acc=1.000, Val F1=0.861                                                                                                                                                     
⏹️ Early stopping після 11 епох
🎉 Найкращий F1: 0.889
wandb:
wandb:                                                                                                                                                                                      
wandb: Run history:
wandb:      epoch ▁▂▂▃▄▅▅▆▇▇█
wandb:   final_f1 ▁
wandb:  train_acc ▁▁▁▃▇▇█████
wandb: train_loss ██▇▆▃▂▂▁▁▁▁
wandb:    val_acc ▁▁▁▆███████
wandb:     val_f1 ▁▁▁▆███████
wandb:   val_loss ███▆▃▂▁▁▁▁▁
wandb:
wandb: Run summary:
wandb:      epoch 10
wandb:   final_f1 0.88889
wandb:  train_acc 1
wandb: train_loss 0.04078
wandb:    val_acc 0.88889
wandb:     val_f1 0.86111
wandb:   val_loss 0.36603
wandb:
wandb:  View run clean-sweep-13 at: https://wandb.ai/msas-agrichain/agri-risk-hyperparameter-search/runs/okcp7r9p
wandb:  View project at: https://wandb.ai/msas-agrichain/agri-risk-hyperparameter-search
wandb: Synced 5 W&B file(s), 0 media file(s), 0 artifact file(s) and 0 other file(s)
wandb: Find logs at: .\wandb\run-20250524_224240-okcp7r9p\logs
wandb: Sweep Agent: Waiting for job.
wandb: Job received.
wandb: Agent Starting Run: qeub38e7 with config:
wandb:  batch_size: 16
wandb:  dropout: 0.15136137946303987
wandb:  learning_rate: 0.0007608857981576607
wandb:  model_name: resnet18
wandb:  num_epochs: 20
wandb:  validation_split: 0.3
wandb:  weight_decay: 0.0045657712627087945
wandb: WARNING Ignoring project 'agri-risk-hyperparameter-search' when running a sweep.
wandb: Tracking run with wandb version 0.19.11
wandb: Run data is saved locally in E:\ml-in-production\HW5_Training_Experiments\pr2\wandb\run-20250524_224444-qeub38e7
wandb: Run `wandb offline` to turn off syncing.
wandb: Syncing run dark-sweep-14
wandb:  View project at https://wandb.ai/msas-agrichain/agri-risk-hyperparameter-search
wandb:  View sweep at https://wandb.ai/msas-agrichain/agri-risk-hyperparameter-search/sweeps/m1p4k5l2
wandb:  View run at https://wandb.ai/msas-agrichain/agri-risk-hyperparameter-search/runs/qeub38e7
💻 Пристрій: cpu
🔄 Створення завантажувачів даних...
🔄 Завантаження даних: diseases
📊 Ризиків: 156, Зображень: 152
✅ Після фільтрації: 86 зображень, 33 класів
📊 Розділення: 60 тренувальних, 26 валідаційних
✅ Готово! Класів: 33, Батчів: train=4, val=2
🧠 Створення моделі resnet18 для 33 класів
🧊 Backbone заморожено
🚀 Початок навчання на 20 епох
Епоха 1: Train Acc=0.050, Val F1=0.103                                                                                                                                                      
Епоха 2: Train Acc=0.167, Val F1=0.038                                                                                                                                                      
Епоха 3: Train Acc=0.317, Val F1=0.077                                                                                                                                                      
🔥 Модель розморожена
Епоха 4: Train Acc=0.450, Val F1=0.519                                                                                                                                                      
Епоха 5: Train Acc=0.900, Val F1=0.731                                                                                                                                                      
Епоха 6: Train Acc=0.950, Val F1=0.679                                                                                                                                                      
Епоха 7: Train Acc=1.000, Val F1=0.747                                                                                                                                                      
Епоха 8: Train Acc=0.983, Val F1=0.788                                                                                                                                                      
Епоха 9: Train Acc=0.983, Val F1=0.708                                                                                                                                                      
Епоха 10: Train Acc=0.917, Val F1=0.750                                                                                                                                                     
Епоха 11: Train Acc=0.983, Val F1=0.451                                                                                                                                                     
Епоха 12: Train Acc=0.950, Val F1=0.641                                                                                                                                                     
Епоха 13: Train Acc=0.917, Val F1=0.618                                                                                                                                                     
⏹️ Early stopping після 13 епох
🎉 Найкращий F1: 0.788
wandb:
wandb:                                                                                                                                                                                      
wandb: Run history:
wandb:      epoch ▁▂▂▃▃▄▅▅▆▆▇▇█
wandb:   final_f1 ▁
wandb:  train_acc ▁▂▃▄▇████▇██▇
wandb: train_loss █▇▆▅▂▁▁▁▁▁▁▁▁
wandb:    val_acc ▂▁▁▆█▇██▇█▆▇▇
wandb:     val_f1 ▂▁▁▅▇▇██▇█▅▇▆
wandb:   val_loss ███▃▂▁▁▁▁▁▅▄▄
wandb:
wandb: Run summary:
wandb:      epoch 12
wandb:   final_f1 0.78846
wandb:  train_acc 0.91667
wandb: train_loss 0.27358
wandb:    val_acc 0.69231
wandb:     val_f1 0.61795
wandb:   val_loss 2.42496
wandb:
wandb:  View run dark-sweep-14 at: https://wandb.ai/msas-agrichain/agri-risk-hyperparameter-search/runs/qeub38e7
wandb:  View project at: https://wandb.ai/msas-agrichain/agri-risk-hyperparameter-search
wandb: Synced 5 W&B file(s), 0 media file(s), 0 artifact file(s) and 0 other file(s)
wandb: Find logs at: .\wandb\run-20250524_224444-qeub38e7\logs
wandb: Agent Starting Run: me7z2o5k with config:
wandb:  batch_size: 16
wandb:  dropout: 0.46080539328501735
wandb:  learning_rate: 0.0014805642983941633
wandb:  model_name: mobilenet_v2
wandb:  num_epochs: 25
wandb:  validation_split: 0.25
wandb:  weight_decay: 2.788960658525523e-05
wandb: WARNING Ignoring project 'agri-risk-hyperparameter-search' when running a sweep.
wandb: Tracking run with wandb version 0.19.11
wandb: Run data is saved locally in E:\ml-in-production\HW5_Training_Experiments\pr2\wandb\run-20250524_224612-me7z2o5k
wandb: Run `wandb offline` to turn off syncing.
wandb: Syncing run comic-sweep-15
wandb:  View project at https://wandb.ai/msas-agrichain/agri-risk-hyperparameter-search
wandb:  View sweep at https://wandb.ai/msas-agrichain/agri-risk-hyperparameter-search/sweeps/m1p4k5l2
wandb:  View run at https://wandb.ai/msas-agrichain/agri-risk-hyperparameter-search/runs/me7z2o5k
💻 Пристрій: cpu
🔄 Створення завантажувачів даних...
🔄 Завантаження даних: diseases
📊 Ризиків: 156, Зображень: 152
✅ Після фільтрації: 86 зображень, 33 класів
📊 Розділення: 64 тренувальних, 22 валідаційних
✅ Готово! Класів: 33, Батчів: train=4, val=2
🧠 Створення моделі mobilenet_v2 для 33 класів
🧊 Backbone заморожено
🚀 Початок навчання на 25 епох
Епоха 1: Train Acc=0.000, Val F1=0.008                                                                                                                                                      
Епоха 2: Train Acc=0.203, Val F1=0.091                                                                                                                                                      
Епоха 3: Train Acc=0.469, Val F1=0.326                                                                                                                                                      
🔥 Модель розморожена
Епоха 4: Train Acc=0.672, Val F1=0.394                                                                                                                                                      
Епоха 5: Train Acc=0.828, Val F1=0.455                                                                                                                                                      
Епоха 6: Train Acc=0.906, Val F1=0.591                                                                                                                                                      
Епоха 7: Train Acc=0.906, Val F1=0.773                                                                                                                                                      
Епоха 8: Train Acc=0.875, Val F1=0.803                                                                                                                                                      
Епоха 9: Train Acc=1.000, Val F1=0.886                                                                                                                                                      
Епоха 10: Train Acc=0.953, Val F1=0.909                                                                                                                                                     
Епоха 11: Train Acc=1.000, Val F1=0.795                                                                                                                                                     
Епоха 12: Train Acc=0.984, Val F1=0.823                                                                                                                                                     
Епоха 13: Train Acc=0.953, Val F1=0.909                                                                                                                                                     
Епоха 14: Train Acc=0.938, Val F1=0.909                                                                                                                                                     
Епоха 15: Train Acc=0.969, Val F1=0.909                                                                                                                                                     
⏹️ Early stopping після 15 епох
🎉 Найкращий F1: 0.909
wandb:
wandb:                                                                                                                                                                                      
wandb: Run history:
wandb:      epoch ▁▁▂▃▃▃▄▅▅▅▆▇▇▇█
wandb:   final_f1 ▁
wandb:  train_acc ▁▂▄▆▇▇▇▇███████
wandb: train_loss █▆▅▄▂▂▂▁▁▁▁▁▁▁▁
wandb:    val_acc ▁▁▄▄▄▆▇▇██▇████
wandb:     val_f1 ▁▂▃▄▄▆▇▇██▇▇███
wandb:   val_loss ██▇▅▅▃▂▁▁▁▂▂▁▁▁
wandb:
wandb: Run summary:
wandb:      epoch 14
wandb:   final_f1 0.90909
wandb:  train_acc 0.96875
wandb: train_loss 0.11341
wandb:    val_acc 0.90909
wandb:     val_f1 0.90909
wandb:   val_loss 0.60327
wandb:
wandb:  View run comic-sweep-15 at: https://wandb.ai/msas-agrichain/agri-risk-hyperparameter-search/runs/me7z2o5k
wandb:  View project at: https://wandb.ai/msas-agrichain/agri-risk-hyperparameter-search
wandb: Synced 5 W&B file(s), 0 media file(s), 0 artifact file(s) and 0 other file(s)
wandb: Find logs at: .\wandb\run-20250524_224612-me7z2o5k\logs
(venv) PS E:\ml-in-production\HW5_Training_Experiments\pr2> 

