#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
import logging
from dataclasses import dataclass

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("data_validation")


@dataclass
class DatasetSchema:
    """
    Схема валидации датасета
    """
    # Обязательные поля
    required_columns: List[str]
    
    # Типы данных для колонок
    column_types: Dict[str, type]
    
    # Ограничения на значения (мин, макс, набор значений и т.д.)
    value_constraints: Dict[str, Dict[str, Union[float, List, Tuple]]]
    
    # Правила обработки пропущенных значений
    missing_value_rules: Dict[str, str]  # 'drop', 'mean', 'median', 'mode', 'constant:value'


class DataValidator:
    """
    Класс для валидации и проверки качества данных
    """
    def __init__(self, schema: DatasetSchema):
        self.schema = schema
        self.validation_results = {}
    
    def validate_dataset(self, df: pd.DataFrame) -> Dict[str, Dict]:
        """
        Валидация датасета по заданной схеме
        """
        results = {
            "is_valid": True,
            "column_presence": {},
            "column_types": {},
            "value_constraints": {},
            "missing_values": {}
        }
        
        # Проверка наличия обязательных колонок
        for col in self.schema.required_columns:
            if col in df.columns:
                results["column_presence"][col] = True
            else:
                results["column_presence"][col] = False
                results["is_valid"] = False
        
        # Если не все обязательные колонки присутствуют, дальнейшая валидация невозможна
        if not all(results["column_presence"].values()):
            return results
        
        # Проверка типов данных
        for col, expected_type in self.schema.column_types.items():
            if col not in df.columns:
                continue
                
            # Проверка типа данных для числовых колонок
            if expected_type in (int, float, np.int64, np.float64):
                is_valid_type = pd.api.types.is_numeric_dtype(df[col])
            # Проверка типа данных для строковых колонок
            elif expected_type == str:
                is_valid_type = pd.api.types.is_string_dtype(df[col])
            # Проверка типа данных для булевых колонок
            elif expected_type == bool:
                is_valid_type = pd.api.types.is_bool_dtype(df[col])
            # Проверка типа данных для datetime колонок
            elif expected_type == pd.Timestamp:
                is_valid_type = pd.api.types.is_datetime64_dtype(df[col])
            else:
                is_valid_type = False
            
            results["column_types"][col] = is_valid_type
            if not is_valid_type:
                results["is_valid"] = False
        
        # Проверка ограничений на значения
        for col, constraints in self.schema.value_constraints.items():
            if col not in df.columns:
                continue
                
            col_constraints = {}
            
            # Проверка минимального значения
            if "min" in constraints:
                min_value = constraints["min"]
                meets_min = df[col].min() >= min_value
                col_constraints["min"] = meets_min
                if not meets_min:
                    results["is_valid"] = False
            
            # Проверка максимального значения
            if "max" in constraints:
                max_value = constraints["max"]
                meets_max = df[col].max() <= max_value
                col_constraints["max"] = meets_max
                if not meets_max:
                    results["is_valid"] = False
            
            # Проверка допустимых значений
            if "allowed_values" in constraints:
                allowed_values = constraints["allowed_values"]
                meets_allowed = df[col].isin(allowed_values).all()
                col_constraints["allowed_values"] = meets_allowed
                if not meets_allowed:
                    results["is_valid"] = False
            
            results["value_constraints"][col] = col_constraints
        
        # Проверка пропущенных значений
        for col in df.columns:
            missing_count = df[col].isna().sum()
            missing_percentage = missing_count / len(df) * 100
            results["missing_values"][col] = {
                "count": missing_count,
                "percentage": missing_percentage
            }
            
            # Проверка правил обработки пропущенных значений
            if col in self.schema.missing_value_rules and missing_count > 0:
                rule = self.schema.missing_value_rules[col]
                if rule == "drop":
                    # Для правила "drop" пропущенные значения недопустимы
                    results["is_valid"] = False
        
        self.validation_results = results
        return results
    
    def validate_and_report(self, df: pd.DataFrame, report_path: Optional[str] = None) -> None:
        """
        Проведение валидации с формированием отчета
        """
        results = self.validate_dataset(df)
        
        # Вывод результатов в лог
        logger.info(f"Dataset validation results: {'PASS' if results['is_valid'] else 'FAIL'}")
        
        # Детальный отчет о проблемах
        if not results["is_valid"]:
            # Отчет о пропущенных колонках
            missing_columns = [col for col, present in results["column_presence"].items() if not present]
            if missing_columns:
                logger.error(f"Missing required columns: {missing_columns}")
            
            # Отчет о некорректных типах данных
            invalid_types = [col for col, valid in results["column_types"].items() if not valid]
            if invalid_types:
                logger.error(f"Columns with invalid data types: {invalid_types}")
            
            # Отчет о нарушениях ограничений на значения
            for col, constraints in results["value_constraints"].items():
                invalid_constraints = [
                    f"{constraint}" for constraint, valid in constraints.items() if not valid
                ]
                if invalid_constraints:
                    logger.error(f"Column '{col}' violates constraints: {invalid_constraints}")
            
            # Отчет о пропущенных значениях
            for col, missing_info in results["missing_values"].items():
                if col in self.schema.missing_value_rules and missing_info["count"] > 0:
                    rule = self.schema.missing_value_rules[col]
                    if rule == "drop":
                        logger.error(
                            f"Column '{col}' has {missing_info['count']} missing values "
                            f"({missing_info['percentage']:.2f}%), but the rule is 'drop'"
                        )
        
        # Сохранение отчета в файл, если указан путь
        if report_path:
            import json
            with open(report_path, "w") as f:
                json.dump(results, f, indent=4)
            logger.info(f"Validation report saved to {report_path}")


def create_agricultural_risk_schema() -> DatasetSchema:
    """
    Создание схемы для валидации датасета с сельскохозяйственными рисками
    """
    return DatasetSchema(
        required_columns=["id", "crop", "risk_description", "risk_type", "severity"],
        column_types={
            "id": int,
            "crop": str,
            "risk_description": str,
            "risk_type": str,
            "severity": str
        },
        value_constraints={
            "risk_type": {"allowed_values": ["diseases", "pests", "weeds"]},
            "severity": {"allowed_values": ["low", "medium", "high"]}
        },
        missing_value_rules={
            "id": "drop",
            "crop": "drop",
            "risk_description": "drop",
            "risk_type": "drop",
            "severity": "drop"
        }
    )


def main():
    """
    Пример использования валидатора данных
    """
    # Создание тестового датасета
    data = {
        "id": [1, 2, 3, 4, 5],
        "crop": ["wheat", "corn", "rice", "potato", "tomato"],
        "risk_description": [
            "Leaf rust causing yellow spots",
            "Corn borer damaging stalks",
            "Rice blast fungus",
            "Colorado potato beetle",
            "Tomato blight"
        ],
        "risk_type": ["diseases", "pests", "diseases", "pests", "diseases"],
        "severity": ["high", "medium", "high", "low", "high"]
    }
    df = pd.DataFrame(data)
    
    # Создание схемы и валидатора
    schema = create_agricultural_risk_schema()
    validator = DataValidator(schema)
    
    # Валидация данных
    validator.validate_and_report(df, "validation_report.json")
    
    # Пример с невалидными данными
    data_invalid = {
        "id": [1, 2, 3, 4, 5],
        "crop": ["wheat", "corn", "rice", "potato", "tomato"],
        "risk_description": [
            "Leaf rust causing yellow spots",
            "Corn borer damaging stalks",
            "Rice blast fungus",
            "Colorado potato beetle",
            None  # Пропущенное значение
        ],
        "risk_type": ["diseases", "pests", "diseases", "pests", "unknown"],  # Недопустимое значение
        "severity": ["high", "medium", "high", "low", "high"]
    }
    df_invalid = pd.DataFrame(data_invalid)
    
    # Валидация невалидных данных
    validator.validate_and_report(df_invalid, "validation_report_invalid.json")


if __name__ == "__main__":
    main()
