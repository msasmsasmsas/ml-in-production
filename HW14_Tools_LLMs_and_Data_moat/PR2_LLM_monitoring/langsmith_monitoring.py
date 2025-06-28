#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Модуль для моніторингу LLM за допомогою LangSmith
"""

import os
import time
import uuid
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv
from langsmith import Client
from langsmith.schemas import Run, RunLike, Example
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.callbacks.tracers.langsmith import LangSmithTracer

# Завантаження змінних середовища
load_dotenv()

# Перевірка наявності ключів API
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "agro_threat_assistant")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Перевірка наявності ключів
if not LANGSMITH_API_KEY:
    raise ValueError("Необхідно встановити змінну середовища LANGSMITH_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("Необхідно встановити змінну середовища OPENAI_API_KEY")

# Ініціалізація клієнта LangSmith
langsmith_client = Client()

# Шаблон системного повідомлення для асистента з загроз сільськогосподарським культурам
SYSTEM_TEMPLATE = """
Ви - асистент агронома, спеціалізований на виявленні та запобіганні загрозам для сільськогосподарських культур. 
Використовуйте свої знання про хвороби рослин, шкідників та бур'яни для надання точної інформації.

Якщо в запиті згадується культура, якої не існує, або ваша відповідь не стосується сільськогосподарських культур, 
поясніть, що ви можете надавати інформацію лише про реальні сільськогосподарські культури та загрози для них.

Відповідайте детально, але лаконічно, українською мовою.
"""

# Створення шаблону запиту для чат-моделі
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_TEMPLATE),
        ("human", "{query}")
    ]
)

# Ініціалізація моделі
llm = ChatOpenAI(temperature=0.7, model="gpt-3.5-turbo", api_key=OPENAI_API_KEY)

# Створення ланцюжка
threat_detection_chain = LLMChain(
    llm=llm,
    prompt=prompt_template,
    verbose=True
)

def create_test_dataset() -> List[Dict[str, str]]:
    """
    Створює тестовий набір даних для LLM

    Returns:
        Список словників з запитами та очікуваними відповідями
    """
    return [
        {
            "query": "Які основні хвороби вражають пшеницю і як їх виявити?",
            "expected": "Хвороби пшениці: борошниста роса, септоріоз, фузаріоз, бура іржа"
        },
        {
            "query": "Що робити, якщо я виявив колорадського жука на картоплі?",
            "expected": "Методи боротьби з колорадським жуком: біологічні, хімічні та механічні"
        },
        {
            "query": "Які бур'яни найбільш шкідливі для кукурудзи?",
            "expected": "Шкідливі бур'яни для кукурудзи: амброзія, пирій, осот, лобода"
        },
        {
            "query": "Як захистити соняшник від склеротинії?",
            "expected": "Методи захисту соняшнику від склеротинії: сівозміна, фунгіциди, стійкі сорти"
        },
        {
            "query": "Які ознаки фітофторозу на помідорах?",
            "expected": "Ознаки фітофторозу: темно-коричневі плями на листках і плодах, загнивання"
        }
    ]

def log_run_to_langsmith(query: str, response: Dict[str, Any], expected: Optional[str] = None) -> Run:
    """
    Логує виконання LLM до LangSmith

    Args:
        query: Запит до LLM
        response: Відповідь від LLM
        expected: Очікувана відповідь (опціонально)

    Returns:
        Об'єкт Run з інформацією про виконання
    """
    # Створюємо трейсер LangSmith
    tracer = LangSmithTracer(project_name=LANGSMITH_PROJECT)

    # Генеруємо унікальний ID для запуску
    run_id = str(uuid.uuid4())

    # Створюємо батьківський запуск
    run = Run(
        id=run_id,
        name="Agro Threat Assistant",
        run_type="chain",
        inputs={"query": query},
        outputs={"response": response["text"]},
        error=None,
        execution_order=1,
        serialized={},
        session_id=None,
        start_time=time.time(),
        end_time=time.time() + 2.0,  # припускаємо, що запуск тривав 2 секунди
        extra={"project": LANGSMITH_PROJECT},
        child_runs=[]
    )

    # Відправляємо запуск до LangSmith
    langsmith_client.create_run(run)

    # Якщо є очікувана відповідь, створюємо приклад для оцінки
    if expected:
        example = Example(
            inputs={"query": query},
            outputs={"expected_response": expected},
            dataset_name="agro_threat_examples"
        )
        langsmith_client.create_example(example)

        # Прив'язуємо запуск до прикладу для порівняння
        langsmith_client.create_feedback(
            run_id,
            "relevance",
            score=0.8,  # симуляція оцінки релевантності
            comment="Автоматична оцінка релевантності"
        )

    return run

def run_with_monitoring(query: str, expected: Optional[str] = None) -> Dict[str, Any]:
    """
    Виконує запит до LLM з моніторингом через LangSmith

    Args:
        query: Запит до LLM
        expected: Очікувана відповідь (опціонально)

    Returns:
        Відповідь від LLM
    """
    # Створюємо трейсер
    tracer = LangSmithTracer(project_name=LANGSMITH_PROJECT)

    # Виконуємо запит з трейсером
    response = threat_detection_chain.run(
        query=query,
        callbacks=[tracer]
    )

    # Формуємо словник відповіді
    result = {"text": response}

    # Додатково логуємо виконання (для демонстрації API)
    log_run_to_langsmith(query, result, expected)

    return result

def evaluate_model_performance(dataset: List[Dict[str, str]]) -> Dict[str, Any]:
    """
    Оцінює продуктивність моделі на наборі тестових запитів

    Args:
        dataset: Список словників з запитами та очікуваними відповідями

    Returns:
        Метрики продуктивності
    """
    results = []
    total_tokens = 0
    total_time = 0

    print("Оцінка продуктивності моделі на тестовому наборі...")

    for i, item in enumerate(dataset):
        print(f"Запит {i+1}/{len(dataset)}: {item['query'][:50]}...")

        # Заміряємо час виконання
        start_time = time.time()

        # Виконуємо запит
        response = run_with_monitoring(item["query"], item.get("expected"))

        # Розраховуємо час виконання
        execution_time = time.time() - start_time
        total_time += execution_time

        # Симуляція підрахунку токенів (в реальному випадку це буде у відповіді API)
        tokens = len(item["query"].split()) + len(response["text"].split())
        total_tokens += tokens

        # Зберігаємо результат
        results.append({
            "query": item["query"],
            "response": response["text"],
            "expected": item.get("expected"),
            "tokens": tokens,
            "time": execution_time
        })

    # Розраховуємо метрики
    metrics = {
        "total_queries": len(dataset),
        "total_tokens": total_tokens,
        "avg_tokens_per_query": total_tokens / len(dataset),
        "total_time": total_time,
        "avg_time_per_query": total_time / len(dataset),
        "results": results
    }

    return metrics

def create_evaluation_dataset() -> None:
    """
    Створює набір даних для оцінки в LangSmith
    """
    dataset = create_test_dataset()

    # Створюємо набір даних в LangSmith
    for item in dataset:
        example = Example(
            inputs={"query": item["query"]},
            outputs={"expected_response": item["expected"]},
            dataset_name="agro_threat_examples"
        )
        langsmith_client.create_example(example)

    print(f"Створено набір даних 'agro_threat_examples' з {len(dataset)} прикладами")

def main():
    """
    Головна функція демонстрації LLM моніторингу
    """
    # Створюємо тестовий набір запитів
    test_dataset = create_test_dataset()

    # Створюємо набір даних для оцінки в LangSmith
    create_evaluation_dataset()

    # Оцінюємо продуктивність моделі
    metrics = evaluate_model_performance(test_dataset)

    # Виводимо загальні метрики
    print("\nЗагальні метрики:")
    print(f"Загальна кількість запитів: {metrics['total_queries']}")
    print(f"Загальна кількість токенів: {metrics['total_tokens']}")
    print(f"Середня кількість токенів на запит: {metrics['avg_tokens_per_query']:.2f}")
    print(f"Загальний час виконання: {metrics['total_time']:.2f} сек")
    print(f"Середній час на запит: {metrics['avg_time_per_query']:.2f} сек")

    # Вивести посилання на проект LangSmith
    print(f"\nРезультати моніторингу доступні в LangSmith: https://smith.langchain.com/project/{LANGSMITH_PROJECT}")

if __name__ == "__main__":
    main()
