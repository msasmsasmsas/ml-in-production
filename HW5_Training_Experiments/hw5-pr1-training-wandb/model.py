# Модуль для визначення моделі

from transformers import AutoModelForSequenceClassification

def get_model(model_name="distilbert-base-uncased", num_labels=3):
    # Завантаження моделі для класифікації послідовностей
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    return model