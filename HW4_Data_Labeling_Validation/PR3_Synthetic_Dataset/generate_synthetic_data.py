import openai
import csv
import os
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
import logging
import pandas as pd
import requests
from typing import List, Dict, Any, Optional
import time
import random

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("synthetic_data_generator")

class ChatGPTDataGenerator:
    """
    Класс для генерации синтетических данных с использованием ChatGPT API
    """
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        self.api_key = api_key
        self.model = model
        self.api_url = "https://api.openai.com/v1/chat/completions"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
    
    def generate_prompt(self, task_type: str, examples: List[Dict[str, Any]]) -> str:
        """
        Генерация промпта для ChatGPT на основе типа задачи и примеров данных
        """
        if task_type == "agricultural_risk":
            prompt = (
                "Сгенерируй синтетические данные для задачи классификации сельскохозяйственных рисков. "
                "Каждая запись должна содержать: идентификатор, название культуры, описание риска, "
                "тип риска (заболевания, вредители или сорняки), степень опасности (низкая, средняя, высокая). "
                "Создай 5 записей в формате JSON.\n\n"
                "Примеры данных:\n"
            )
            
            for example in examples:
                prompt += json.dumps(example, ensure_ascii=False) + "\n"
        
        elif task_type == "crop_recommendation":
            prompt = (
                "Сгенерируй синтетические данные для задачи рекомендации сельскохозяйственных культур. "
                "Каждая запись должна содержать: тип почвы, уровень pH, содержание азота, фосфора и калия, "
                "среднюю температуру, влажность, осадки и рекомендуемую культуру. "
                "Создай 5 записей в формате JSON.\n\n"
                "Примеры данных:\n"
            )
            
            for example in examples:
                prompt += json.dumps(example, ensure_ascii=False) + "\n"
        
        else:
            prompt = (
                f"Сгенерируй синтетические данные для задачи {task_type}. "
                "Создай 5 записей в формате JSON на основе следующих примеров:\n\n"
            )
            
            for example in examples:
                prompt += json.dumps(example, ensure_ascii=False) + "\n"
        
        return prompt
    
    def generate_data(self, prompt: str, temperature: float = 0.7, max_tokens: int = 2000) -> List[Dict[str, Any]]:
        """
        Запрос к ChatGPT API для генерации данных
        """
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        try:
            response = requests.post(self.api_url, headers=self.headers, json=payload)
            response.raise_for_status()
            
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            
            # Попытка извлечь JSON из ответа
            json_data = self._extract_json_from_text(content)
            
            return json_data
        
        except requests.exceptions.RequestException as e:
            logger.error(f"Ошибка при запросе к API: {e}")
            if hasattr(e, 'response') and e.response:
                logger.error(f"Детали ошибки: {e.response.text}")
            return []
        
        except Exception as e:
            logger.error(f"Непредвиденная ошибка: {e}")
            return []
    
    def _extract_json_from_text(self, text: str) -> List[Dict[str, Any]]:
        """
        Извлечение JSON данных из текстового ответа
        """
        # Попытка найти и спарсить все JSON объекты в тексте
        json_data = []
        lines = text.split("\n")
        
        # Поиск строк, которые могут содержать JSON
        json_lines = []
        current_json = ""
        
        for line in lines:
            line = line.strip()
            if line.startswith("{") or current_json:
                current_json += line
                if line.endswith("}"):
                    json_lines.append(current_json)
                    current_json = ""
        
        # Парсинг найденных JSON объектов
        for json_str in json_lines:
            try:
                data = json.loads(json_str)
                json_data.append(data)
            except json.JSONDecodeError:
                continue
        
        # Если не удалось найти JSON объекты, попробуем извлечь JSON массив
        if not json_data and "[" in text and "]" in text:
            try:
                start_idx = text.find("[")
                end_idx = text.rfind("]") + 1
                json_array = text[start_idx:end_idx]
                json_data = json.loads(json_array)
            except json.JSONDecodeError:
                pass
        
        return json_data

def load_examples(file_path: str) -> List[Dict[str, Any]]:
    """
    Загрузка примеров данных из файла JSON или CSV
    """
    if not os.path.exists(file_path):
        logger.error(f"Файл не найден: {file_path}")
        return []
    
    try:
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == '.json':
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        elif file_ext == '.csv':
            df = pd.read_csv(file_path)
            return df.to_dict('records')
        
        else:
            logger.error(f"Неподдерживаемый формат файла: {file_ext}")
            return []
    
    except Exception as e:
        logger.error(f"Ошибка при загрузке примеров: {e}")
        return []

def save_data(data: List[Dict[str, Any]], output_file: str):
    """
    Сохранение сгенерированных данных в файл
    """
    try:
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        
        file_ext = os.path.splitext(output_file)[1].lower()
        
        if file_ext == '.json':
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        
        elif file_ext == '.csv':
            df = pd.DataFrame(data)
            df.to_csv(output_file, index=False, encoding='utf-8')
        
        else:
            logger.error(f"Неподдерживаемый формат выходного файла: {file_ext}")
            return
        
        logger.info(f"Данные сохранены в файл: {output_file}")
    
    except Exception as e:
        logger.error(f"Ошибка при сохранении данных: {e}")

def generate_batch(generator: ChatGPTDataGenerator, task_type: str, examples: List[Dict[str, Any]], 
                  batch_size: int, temperature: float) -> List[Dict[str, Any]]:
    """
    Генерация нескольких батчей данных
    """
    all_data = []
    
    for i in range(batch_size):
        logger.info(f"Генерация батча {i+1}/{batch_size}")
        
        # Случайные примеры для разнообразия
        random_examples = random.sample(examples, min(3, len(examples)))
        
        prompt = generator.generate_prompt(task_type, random_examples)
        batch_data = generator.generate_data(prompt, temperature)
        
        if batch_data:
            all_data.extend(batch_data)
            logger.info(f"Сгенерировано {len(batch_data)} записей")
        else:
            logger.warning("Не удалось сгенерировать данные в этом батче")
        
        # Пауза между запросами, чтобы не превысить лимиты API
        if i < batch_size - 1:
            time.sleep(2)
    
    return all_data

def main():
    parser = argparse.ArgumentParser(description="Генерация синтетических данных с использованием ChatGPT")
    parser.add_argument('--api_key', type=str, help='API ключ для OpenAI API')
    parser.add_argument('--model', type=str, default="gpt-3.5-turbo", help='Модель ChatGPT для использования')
    parser.add_argument('--examples_file', type=str, required=True, help='Путь к файлу с примерами данных (JSON или CSV)')
    parser.add_argument('--output_file', type=str, required=True, help='Путь для сохранения сгенерированных данных')
    parser.add_argument('--task_type', type=str, default="agricultural_risk", 
                        choices=["agricultural_risk", "crop_recommendation", "custom"], 
                        help='Тип задачи для генерации данных')
    parser.add_argument('--temperature', type=float, default=0.7, help='Параметр temperature для ChatGPT (0.0-1.0)')
    parser.add_argument('--batch_size', type=int, default=3, help='Количество батчей для генерации')
    
    args = parser.parse_args()
    
    # Получение API ключа из аргументов или переменных окружения
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.error("API ключ не указан. Используйте --api_key или установите переменную окружения OPENAI_API_KEY")
        return
    
    # Загрузка примеров данных
    examples = load_examples(args.examples_file)
    if not examples:
        logger.error("Не удалось загрузить примеры данных")
        return
    
    logger.info(f"Загружено {len(examples)} примеров данных")
    
    # Инициализация генератора данных
    generator = ChatGPTDataGenerator(api_key, args.model)
    
    # Генерация данных
    generated_data = generate_batch(generator, args.task_type, examples, args.batch_size, args.temperature)
    
    if generated_data:
        logger.info(f"Всего сгенерировано {len(generated_data)} записей")
        save_data(generated_data, args.output_file)
    else:
        logger.error("Не удалось сгенерировать данные")

if __name__ == "__main__":
    main()
openai.api_key = os.getenv("OPENAI_API_KEY")

def generate_review():
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Generate a short product review (positive or negative)."},
        ]
    )
    return response.choices[0].message["content"]

def save_to_csv(reviews, filename="synthetic_dataset.csv"):
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "text"])
        for i, review in enumerate(reviews, 1):
            writer.writerow([i, review])

if __name__ == "__main__":
    reviews = [generate_review() for _ in range(10)]
    save_to_csv(reviews)