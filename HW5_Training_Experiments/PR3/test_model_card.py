#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Швидкий тест для PR3: Model Card
Мінімальний приклад створення Model Card
"""

import os
import json


import os

def create_quick_model_card():
    """Швидке створення Model Card без зайвих залежностей"""

    print("🚀 Швидке створення Model Card...")

    try:
        from model_card import ModelCard

        # Створення простої карти
        card = ModelCard("MobileNetV2 Agricultural Classifier")

        # Основна інформація
        card.set_model_details(
            version="1.0",
            architecture="MobileNetV2 з 33 класами для хвороб рослин",
            developers="AgriTech Team",
            license_info="MIT License"
        )

        # Призначення
        card.set_intended_use(
            primary_use="Класифікація хвороб сільськогосподарських рослин",
            primary_users="Фермери та агрономи",
            limitations="Працює тільки з фотографіями листя та плодів"
        )

        # Дані для навчання
        card.set_training_data(
            dataset_description="86 зображень хвороб рослин, 33 класи",
            data_preprocessing="Зміна розміру, нормалізація, аугментація",
            data_sources=["Внутрішня база даних"],
            data_collection_timeframe="2024",
            data_size={
                "Загалом": 86,
                "Тренувальних": 60,
                "Валідаційних": 26
            }
        )

        # Результати
        card.set_evaluation_data(
            dataset_description="Валідаційна вибірка",
            evaluation_results={
                "F1-скор": 0.827,
                "Accuracy": 0.846,
                "Епох навчання": 18
            }
        )

        # Метрики
        card.add_quantitative_analysis(
            metrics={
                "F1-скор": 0.827,
                "Точність": 0.846,
                "Класів": 33
            }
        )

        # Етика
        card.set_ethical_considerations(
            risks_and_harms="Неправильна діагностика може призвести до економічних втрат",
            use_cases_to_avoid="Не використовувати для критичних рішень без експерта"
        )

        # Застереження
        card.set_caveats_recommendations(
            known_caveats="Може не працювати при поганому освітленні",
            recommendations="Використовувати з консультацією агронома"
        )

        # Експорт
        os.makedirs("cards", exist_ok=True)

        markdown_file = "cards/quick_model_card.md"
        html_file = "cards/quick_model_card.html"
        json_file = "cards/quick_model_card.json"

        card.to_markdown(markdown_file)
        card.to_html(html_file)
        card.to_json(json_file)

        print("✅ Model Card створено:")
        print(f"  📝 {markdown_file}")
        print(f"  🌐 {html_file}")
        print(f"  📊 {json_file}")

        # Перевірка вмісту
        with open(markdown_file, 'r', encoding='utf-8') as f:
            content = f.read()

        if "F1-скор" in content and "0.827" in content:
            print("✅ Метрики правильно включені")
        else:
            print("⚠️ Можливо, проблема з метриками")

        if len(content) > 2000:
            print("✅ Достатньо деталей у карті")
        else:
            print("⚠️ Карта може бути занадто короткою")

        return True

    except Exception as e:
        print(f"❌ Помилка: {e}")
        import traceback
        traceback.print_exc()
        return False


def create_metadata_example():
    """Створення прикладу метаданих"""

    print("\n📝 Створення прикладу метаданих...")

    metadata = {
        "model_name": "mobilenet_v2",
        "num_classes": 33,
        "class_names": [
                           "Альтернариоз", "Фузариоз", "Мучниста роса", "Септориоз",
                           "Ржавчина", "Антракноз", "Фітофтороз", "Пероноспороз",
                           "Церкоспороз", "Бактеріальний ожог"
                       ][:10],  # Перші 10 для прикладу
        "best_f1": 0.827,
        "config": {
            "batch_size": 8,
            "learning_rate": 0.001,
            "num_epochs": 18
        },
        "dataset_size": {
            "total": 86,
            "train": 60,
            "val": 26
        }
    }

    os.makedirs("models", exist_ok=True)
    metadata_file = "models/example_metadata.json"

    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"✅ Метадані створено: {metadata_file}")
    return metadata_file


def test_with_metadata():
    """Тест з автоматичним створенням з метаданих"""

    print("\n🧪 Тест автоматичного створення...")

    try:
        from model_card import create_model_card_from_metadata

        # Створюємо метадані
        metadata_file = create_metadata_example()

        # Створюємо карту з метаданих
        card_paths = create_model_card_from_metadata(metadata_file, "cards")

        if card_paths:
            print("✅ Автоматичне створення працює")
            return True
        else:
            print("❌ Проблема з автоматичним створенням")
            return False

    except Exception as e:
        print(f"❌ Помилка автоматичного створення: {e}")
        return False


def main():
    """Основна функція швидкого тесту"""

    print("⚡ ШВИДКИЙ ТЕСТ PR3: MODEL CARD")
    print("=" * 40)

    success_count = 0

    # Тест 1: Ручне створення
    print("\n1️⃣ Тест ручного створення...")
    if create_quick_model_card():
        success_count += 1

    # Тест 2: Автоматичне створення
    print("\n2️⃣ Тест автоматичного створення...")
    if test_with_metadata():
        success_count += 1

    # Результат
    print(f"\n📊 Результат: {success_count}/2 тестів пройдено")

    if success_count == 2:
        print("🎉 ВСЕ ПРАЦЮЄ! PR3 готовий!")
        print("\n📁 Створені файли:")
        print("  cards/quick_model_card.html - відкрийте у браузері")
        print("  cards/quick_model_card.md - для GitHub")
        print("  models/example_metadata.json - приклад метаданих")
    else:
        print("⚠️ Є проблеми, перевірте помилки вище")

    return success_count == 2


if __name__ == "__main__":
    main()