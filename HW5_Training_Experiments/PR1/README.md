# PR1: Навчання моделі з логуванням у W&B

Цей модуль реалізує навчання моделі класифікації сільськогосподарських ризиків з використанням Weights & Biases (W&B) для відстеження експериментів.

## Функціональність

- Завантаження та підготовка датасету сільськогосподарських ризиків
- Навчання CNN моделі (ResNet18, ResNet50 або MobileNetV2)
- Логування метрик та артефактів у W&B
- Збереження найкращої моделі та метаданих

## Структура

- `train.py` - основний скрипт навчання моделі з інтеграцією W&B
- `models/` - директорія для збереження навчених моделей
- `README.md` - документація

## Використання

1. Встановіть необхідні залежності:


------------------------------------------------------------
Епоха 22/50
Навчання епоха 22: 100%|██████████| 2/2 [00:20<00:00, 10.45s/it, loss=1.13, acc=0.969, lr=5e-5]
Валідація: 100%|██████████| 1/1 [00:18<00:00, 18.19s/it]
Навчання епоха 23:   0%|          | 0/2 [00:00<?, ?it/s]Train Loss: 1.1140, Train Acc: 0.9688
Val Loss: 1.8100, Val Acc: 0.6000, Val F1: 0.6000
LR: 5.00e-05
------------------------------------------------------------
Епоха 23/50
Навчання епоха 23: 100%|██████████| 2/2 [00:20<00:00, 10.34s/it, loss=1.19, acc=0.938, lr=5e-5]
Валідація: 100%|██████████| 1/1 [00:18<00:00, 18.10s/it]
Train Loss: 1.2431, Train Acc: 0.9375
Val Loss: 1.7624, Val Acc: 0.6000, Val F1: 0.6000
LR: 5.00e-05
Early stopping після 23 епох

🎉 Навчання завершено!
Найкращий валідаційний F1: 0.6000
wandb: uploading history steps 22-22, summary, console lines 215-222
wandb:                                                                                
wandb: 
wandb: Run history:
wandb:      epoch ▁▁▂▂▂▃▃▃▄▄▄▅▅▅▅▆▆▆▇▇▇██
wandb:         lr ██████████████████▁▁▁▁▁
wandb:  train_acc ▁▂▂▁▁▁▂▂▄▄▅▆▆▆▇▇▇▇▇▇▇██
wandb: train_loss ██████▇▆▆▅▅▄▄▃▃▃▂▂▂▂▂▁▁
wandb:    val_acc ▁▁▁▂▂▂▃▅▆▆▆▆███████████
wandb:     val_f1 ▁▁▁▁▁▁▃▄▅▆▆▆███████████
wandb:   val_loss █████▇▇▇▆▆▅▅▄▄▄▃▃▂▂▂▂▁▁
wandb: 
wandb: Run summary:
wandb:      epoch 22
wandb:         lr 5e-05
wandb:  train_acc 0.9375
wandb: train_loss 1.24306
wandb:    val_acc 0.6
wandb:     val_f1 0.6
wandb:   val_loss 1.76242
wandb: 
wandb:  View run sleek-dust-1 at: https://wandb.ai/msas-agrichain/agri-risk-classification-improved/runs/mkufp1fe
wandb:  View project at: https://wandb.ai/msas-agrichain/agri-risk-classification-improved
wandb: Synced 5 W&B file(s), 0 media file(s), 0 artifact file(s) and 0 other file(s)
wandb: Find logs at: .\wandb\run-20250524_200154-mkufp1fe\logs


Епоха 17/30
Епоха 17: 100%|██████████| 8/8 [00:06<00:00,  1.31it/s, loss=0.569, acc=0.967, lr=0.0005]
Епоха 18:   0%|          | 0/8 [00:00<?, ?it/s]Train: Loss=0.582, Acc=0.967
Val: Loss=1.182, Acc=0.846, F1=0.821

Епоха 18/30
Епоха 18: 100%|██████████| 8/8 [00:05<00:00,  1.41it/s, loss=0.499, acc=1, lr=0.0005]
Train: Loss=0.481, Acc=1.000
Val: Loss=1.224, Acc=0.846, F1=0.801
⏹️ Early stopping після 18 епох

🎉 Швидке навчання завершено!
Найкращий F1: 0.827
🎯 ДОБРЕ! Модель навчилася
💡 Для покращення спробуйте:
  - Збільшити кількість даних
  - Використати більші моделі
  - Додати аугментації
wandb: uploading wandb-summary.json; uploading config.yaml
wandb:                                                                                
wandb: 
wandb: Run history:
wandb:      epoch ▁▁▂▂▃▃▃▄▄▅▅▆▆▆▇▇██
wandb:         lr ██████████████▁▁▁▁
wandb:  train_acc ▁▃▅▆▆▇▇▇▇▇▇▇▇█▇███
wandb: train_loss █▆▅▄▂▂▂▂▂▂▂▁▂▁▁▁▁▁
wandb:    val_acc ▁▃▄▅▇█████████▇███
wandb:     val_f1 ▁▃▄▄▇██▇██████▇███
wandb:   val_loss █▇▅▄▂▂▂▁▁▁▁▂▁▁▂▁▁▁
wandb: 
wandb: Run summary:
wandb:      epoch 17
wandb:         lr 0.0005
wandb:  train_acc 1
wandb: train_loss 0.4808
wandb:    val_acc 0.84615
wandb:     val_f1 0.80128
wandb:   val_loss 1.22441
wandb: 
wandb:  View run driven-bee-2 at: https://wandb.ai/msas-agrichain/agri-risk-classification-fast/runs/oikwnqjg
wandb:  View project at: https://wandb.ai/msas-agrichain/agri-risk-classification-fast
wandb: Synced 5 W&B file(s), 0 media file(s), 0 artifact file(s) and 0 other file(s)
wandb: Find logs at: .\wandb\run-20250524_214617-oikwnqjg\logs
