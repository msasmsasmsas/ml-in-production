# Скрипт для створення картки моделі

from model_card_toolkit import ModelCardToolkit, model_card

def create_model_card():
    # Ініціалізація інструментарію для створення картки
    mct = ModelCardToolkit("./model_card_output")

    # Створення картки моделі
    mc = model_card.ModelCard()
    mc.model_details.name = "DistilBERT for Agricultural Threat Classification"
    mc.model_details.overview = (
        "Ця модель класифікує тексти, що описують загрози сільськогосподарським культурам, на три категорії: бур'яни, шкідники, хвороби."
    )
    mc.model_details.owners = [{"name": "Your Name", "contact": "your.email@example.com"}]
    mc.model_details.version = {"name": "v1.0", "date": "2025-05-12"}
    mc.model_details.references = [
        {"reference": "Hugging Face Transformers: https://huggingface.co/docs/transformers"}
    ]
    mc.model_details.path = "./trained_model"

    # Метрики продуктивності
    mc.quantitative_analysis.performance_metrics = [
        {"type": "accuracy", "value": 0.85, "slice": "test_set"}
    ]

    # Дані для тренування
    mc.training_data.description = "Синтетичний датасет україномовних текстів, що описують загрози сільськогосподарським культурам."
    mc.training_data.size = 1000
    mc.evaluation_data.description = "20% від тренувального датасету."

    # Збереження картки
    mct.update_model_card(mc)
    mct.export_format(output_dir="./model_card_output", template="markdown")

if __name__ == "__main__":
    create_model_card()