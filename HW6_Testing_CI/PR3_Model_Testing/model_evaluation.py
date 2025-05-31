#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Union, Tuple, Any, Callable
import logging
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    precision_recall_curve, average_precision_score,
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
import joblib

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("model_evaluation")


class ModelEvaluator:
    """
    Класс для комплексной оценки качества моделей машинного обучения
    """
    def __init__(self, model_path: str, model_name: str, task_type: str = "classification"):
        """
        Инициализация оценщика моделей
        
        Args:
            model_path: Путь к сохраненной модели
            model_name: Название модели
            task_type: Тип задачи ('classification' или 'regression')
        """
        self.model_path = model_path
        self.model_name = model_name
        self.task_type = task_type
        self.model = self.load_model(model_path)
        self.evaluation_results = {}
    
    def load_model(self, model_path: str) -> Any:
        """
        Загрузка модели из файла
        
        Args:
            model_path: Путь к файлу модели
            
        Returns:
            Загруженная модель
        """
        try:
            return joblib.load(model_path)
        except Exception as e:
            logger.error(f"Error loading model from {model_path}: {e}")
            raise
    
    def evaluate_classification(self, X: np.ndarray, y_true: np.ndarray, 
                              class_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Оценка качества модели классификации
        
        Args:
            X: Признаки
            y_true: Истинные метки классов
            class_names: Названия классов
            
        Returns:
            Словарь с метриками качества
        """
        # Предсказания моделей
        y_pred = self.model.predict(X)
        
        # Для метрик, требующих вероятностей классов
        if hasattr(self.model, "predict_proba"):
            y_prob = self.model.predict_proba(X)
            if y_prob.shape[1] == 2:  # Бинарная классификация
                y_prob = y_prob[:, 1]
        else:
            y_prob = None
        
        # Рассчитываем базовые метрики
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average='weighted'),
            "recall": recall_score(y_true, y_pred, average='weighted'),
            "f1_score": f1_score(y_true, y_pred, average='weighted')
        }
        
        # Добавляем метрики для бинарной классификации, если применимо
        if len(np.unique(y_true)) == 2 and y_prob is not None:
            metrics.update({
                "roc_auc": roc_auc_score(y_true, y_prob),
                "average_precision": average_precision_score(y_true, y_prob)
            })
        
        # Матрица ошибок
        cm = confusion_matrix(y_true, y_pred)
        metrics["confusion_matrix"] = cm
        
        # Детальный отчет по классам
        if class_names is None:
            class_names = [str(i) for i in range(len(np.unique(y_true)))]
        
        report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
        metrics["classification_report"] = report
        
        # Сохраняем результаты
        self.evaluation_results = metrics
        
        return metrics
    
    def evaluate_regression(self, X: np.ndarray, y_true: np.ndarray) -> Dict[str, Any]:
        """
        Оценка качества модели регрессии
        
        Args:
            X: Признаки
            y_true: Истинные значения целевой переменной
            
        Returns:
            Словарь с метриками качества
        """
        # Предсказания модели
        y_pred = self.model.predict(X)
        
        # Рассчитываем метрики
        metrics = {
            "mse": mean_squared_error(y_true, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
            "mae": mean_absolute_error(y_true, y_pred),
            "r2": r2_score(y_true, y_pred),
            "explained_variance": np.var(y_pred) / np.var(y_true) if np.var(y_true) > 0 else 0
        }
        
        # Сохраняем результаты и предсказания
        metrics["predictions"] = y_pred
        self.evaluation_results = metrics
        
        return metrics
    
    def evaluate(self, X: np.ndarray, y_true: np.ndarray, 
                class_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Общий метод оценки качества модели в зависимости от типа задачи
        
        Args:
            X: Признаки
            y_true: Истинные значения целевой переменной
            class_names: Названия классов (для классификации)
            
        Returns:
            Словарь с метриками качества
        """
        if self.task_type == "classification":
            return self.evaluate_classification(X, y_true, class_names)
        elif self.task_type == "regression":
            return self.evaluate_regression(X, y_true)
        else:
            raise ValueError(f"Unsupported task type: {self.task_type}")
    
    def cross_validate(self, X: np.ndarray, y: np.ndarray, cv: int = 5, 
                      scoring: str = 'accuracy') -> Dict[str, Any]:
        """
        Кросс-валидация модели
        
        Args:
            X: Признаки
            y: Истинные значения целевой переменной
            cv: Количество фолдов для кросс-валидации
            scoring: Метрика для оценки
            
        Returns:
            Словарь с результатами кросс-валидации
        """
        # Выбираем схему разбиения в зависимости от типа задачи
        if self.task_type == "classification":
            cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        else:
            cv_splitter = cv
        
        # Проводим кросс-валидацию
        cv_scores = cross_val_score(self.model, X, y, cv=cv_splitter, scoring=scoring)
        
        # Формируем результаты
        cv_results = {
            "mean_score": np.mean(cv_scores),
            "std_score": np.std(cv_scores),
            "min_score": np.min(cv_scores),
            "max_score": np.max(cv_scores),
            "all_scores": cv_scores.tolist()
        }
        
        return cv_results
    
    def visualize_classification_results(self, X: np.ndarray, y_true: np.ndarray, 
                                       class_names: Optional[List[str]] = None,
                                       output_dir: str = "model_evaluation") -> None:
        """
        Визуализация результатов оценки модели классификации
        
        Args:
            X: Признаки
            y_true: Истинные метки классов
            class_names: Названия классов
            output_dir: Директория для сохранения визуализаций
        """
        if not self.evaluation_results:
            self.evaluate_classification(X, y_true, class_names)
        
        # Создаем директорию для выходных файлов, если ее нет
        os.makedirs(output_dir, exist_ok=True)
        
        # Предсказания модели
        y_pred = self.model.predict(X)
        
        # Для метрик, требующих вероятностей классов
        if hasattr(self.model, "predict_proba"):
            y_prob = self.model.predict_proba(X)
        else:
            y_prob = None
        
        # Если названия классов не указаны, используем номера классов
        if class_names is None:
            class_names = [str(i) for i in range(len(np.unique(y_true)))]
        
        # 1. Матрица ошибок
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y_true, y_pred)
        
        # Нормализуем матрицу ошибок
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        sns.heatmap(cm_norm, annot=cm, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.title(f'Confusion Matrix - {self.model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{self.model_name}_confusion_matrix.png"))
        plt.close()
        
        # 2. Отчет о классификации
        metrics = self.evaluation_results
        plt.figure(figsize=(12, 6))
        
        # Собираем метрики по классам
        class_metrics = []
        for cls in class_names:
            if cls in metrics["classification_report"]:
                class_dict = metrics["classification_report"][cls]
                class_metrics.append({
                    "class": cls,
                    "precision": class_dict["precision"],
                    "recall": class_dict["recall"],
                    "f1-score": class_dict["f1-score"],
                    "support": class_dict["support"]
                })
        
        # Преобразуем в DataFrame для визуализации
        df_metrics = pd.DataFrame(class_metrics)
        
        # Строим график метрик по классам
        df_plot = df_metrics.melt(id_vars=['class', 'support'], 
                                value_vars=['precision', 'recall', 'f1-score'],
                                var_name='metric', value_name='value')
        
        plt.figure(figsize=(12, 6))
        sns.barplot(x='class', y='value', hue='metric', data=df_plot)
        plt.title(f'Classification Metrics by Class - {self.model_name}')
        plt.ylabel('Score')
        plt.xlabel('Class')
        plt.ylim(0, 1.0)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{self.model_name}_class_metrics.png"))
        plt.close()
        
        # 3. ROC-кривая для бинарной классификации
        if len(class_names) == 2 and y_prob is not None:
            from sklearn.metrics import roc_curve, auc
            
            plt.figure(figsize=(8, 8))
            y_prob_binary = y_prob[:, 1] if y_prob.shape[1] == 2 else y_prob
            
            fpr, tpr, _ = roc_curve(y_true, y_prob_binary)
            roc_auc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, color='darkorange', lw=2,
                    label=f'ROC curve (area = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve - {self.model_name}')
            plt.legend(loc="lower right")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{self.model_name}_roc_curve.png"))
            plt.close()
            
            # 4. Precision-Recall кривая
            plt.figure(figsize=(8, 8))
            precision, recall, _ = precision_recall_curve(y_true, y_prob_binary)
            avg_precision = average_precision_score(y_true, y_prob_binary)
            
            plt.plot(recall, precision, color='blue', lw=2,
                    label=f'Precision-Recall curve (AP = {avg_precision:.2f})')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.ylim([0.0, 1.05])
            plt.xlim([0.0, 1.0])
            plt.title(f'Precision-Recall Curve - {self.model_name}')
            plt.legend(loc="lower left")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{self.model_name}_precision_recall_curve.png"))
            plt.close()
    
    def visualize_regression_results(self, X: np.ndarray, y_true: np.ndarray,
                                    output_dir: str = "model_evaluation") -> None:
        """
        Визуализация результатов оценки модели регрессии
        
        Args:
            X: Признаки
            y_true: Истинные значения целевой переменной
            output_dir: Директория для сохранения визуализаций
        """
        if not self.evaluation_results:
            self.evaluate_regression(X, y_true)
        
        # Создаем директорию для выходных файлов, если ее нет
        os.makedirs(output_dir, exist_ok=True)
        
        # Предсказания модели
        y_pred = self.model.predict(X)
        
        # 1. График истинных значений vs предсказания
        plt.figure(figsize=(10, 8))
        plt.scatter(y_true, y_pred, alpha=0.5)
        
        # Добавляем линию идеального предсказания
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        
        plt.title(f'True vs Predicted - {self.model_name}')
        plt.xlabel('True Values')
        plt.ylabel('Predicted Values')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{self.model_name}_true_vs_pred.png"))
        plt.close()
        
        # 2. Распределение ошибок
        errors = y_true - y_pred
        
        plt.figure(figsize=(10, 6))
        sns.histplot(errors, kde=True)
        plt.title(f'Error Distribution - {self.model_name}')
        plt.xlabel('Prediction Error')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{self.model_name}_error_distribution.png"))
        plt.close()
        
        # 3. Ошибки vs предсказания (для проверки гомоскедастичности)
        plt.figure(figsize=(10, 6))
        plt.scatter(y_pred, errors, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.title(f'Prediction Errors vs Predicted Values - {self.model_name}')
        plt.xlabel('Predicted Values')
        plt.ylabel('Prediction Errors')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{self.model_name}_error_vs_predicted.png"))
        plt.close()
        
        # 4. Сводка метрик
        metrics = self.evaluation_results
        metrics_to_plot = ["mse", "rmse", "mae", "r2", "explained_variance"]
        values = [metrics.get(m, 0) for m in metrics_to_plot]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(metrics_to_plot, values, color='skyblue')
        
        # Добавляем значения над столбцами
        for bar, val in zip(bars, values):
            if val < 0.01:  # Для очень маленьких значений используем научную нотацию
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{val:.2e}', ha='center')
            else:
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{val:.3f}', ha='center')
        
        plt.title(f'Regression Metrics - {self.model_name}')
        plt.ylabel('Value')
        plt.grid(axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{self.model_name}_regression_metrics.png"))
        plt.close()
    
    def visualize_results(self, X: np.ndarray, y_true: np.ndarray,
                         class_names: Optional[List[str]] = None,
                         output_dir: str = "model_evaluation") -> None:
        """
        Визуализация результатов оценки модели в зависимости от типа задачи
        
        Args:
            X: Признаки
            y_true: Истинные значения целевой переменной
            class_names: Названия классов (для классификации)
            output_dir: Директория для сохранения визуализаций
        """
        if self.task_type == "classification":
            self.visualize_classification_results(X, y_true, class_names, output_dir)
        elif self.task_type == "regression":
            self.visualize_regression_results(X, y_true, output_dir)
        else:
            raise ValueError(f"Unsupported task type: {self.task_type}")
    
    def generate_report(self, output_path: Optional[str] = None) -> str:
        """
        Генерация текстового отчета об оценке модели
        
        Args:
            output_path: Путь для сохранения отчета (если не указан, отчет только возвращается)
            
        Returns:
            Текстовый отчет о качестве модели
        """
        if not self.evaluation_results:
            logger.warning("No evaluation results available. Run evaluate() first.")
            return "No evaluation results available."
        
        # Формируем отчет в зависимости от типа задачи
        if self.task_type == "classification":
            report = self._generate_classification_report()
        elif self.task_type == "regression":
            report = self._generate_regression_report()
        else:
            return f"Unsupported task type: {self.task_type}"
        
        # Сохраняем отчет в файл, если указан путь
        if output_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            with open(output_path, "w") as f:
                f.write(report)
            logger.info(f"Evaluation report saved to {output_path}")
        
        return report
    
    def _generate_classification_report(self) -> str:
        """
        Генерация отчета для модели классификации
        
        Returns:
            Текстовый отчет
        """
        metrics = self.evaluation_results
        
        report = f"MODEL EVALUATION REPORT - {self.model_name}\n"
        report += "=" * 60 + "\n\n"
        report += "Task Type: Classification\n"
        report += f"Model Path: {self.model_path}\n\n"
        
        report += "SUMMARY METRICS\n"
        report += "-" * 60 + "\n"
        report += f"Accuracy:     {metrics.get('accuracy', 'N/A'):.4f}\n"
        report += f"Precision:    {metrics.get('precision', 'N/A'):.4f}\n"
        report += f"Recall:       {metrics.get('recall', 'N/A'):.4f}\n"
        report += f"F1 Score:     {metrics.get('f1_score', 'N/A'):.4f}\n"
        
        if "roc_auc" in metrics:
            report += f"ROC AUC:      {metrics.get('roc_auc', 'N/A'):.4f}\n"
        
        if "average_precision" in metrics:
            report += f"Avg Precision: {metrics.get('average_precision', 'N/A'):.4f}\n"
        
        report += "\nDETAILED CLASSIFICATION REPORT\n"
        report += "-" * 60 + "\n"
        
        # Добавляем детальный отчет по классам, если доступен
        if "classification_report" in metrics:
            class_report = metrics["classification_report"]
            
            # Заголовок
            report += f"{'Class':<15} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}\n"
            report += "-" * 60 + "\n"
            
            # Метрики по каждому классу
            for cls, values in class_report.items():
                if cls in ["accuracy", "macro avg", "weighted avg", "samples avg"]:
                    continue
                
                report += f"{cls:<15} {values['precision']:<10.4f} {values['recall']:<10.4f} "
                report += f"{values['f1-score']:<10.4f} {values['support']:<10}\n"
            
            # Средние значения
            report += "-" * 60 + "\n"
            for avg_type in ["macro avg", "weighted avg"]:
                if avg_type in class_report:
                    report += f"{avg_type:<15} {class_report[avg_type]['precision']:<10.4f} "
                    report += f"{class_report[avg_type]['recall']:<10.4f} "
                    report += f"{class_report[avg_type]['f1-score']:<10.4f} "
                    report += f"{class_report[avg_type]['support']:<10}\n"
        
        report += "\nCONFUSION MATRIX\n"
        report += "-" * 60 + "\n"
        
        if "confusion_matrix" in metrics:
            cm = metrics["confusion_matrix"]
            # Форматируем матрицу ошибок для текстового отчета
            cm_str = np.array2string(cm, separator=', ')
            report += cm_str + "\n"
        
        report += "\n" + "=" * 60 + "\n"
        report += f"End of report for {self.model_name}\n"
        
        return report
    
    def _generate_regression_report(self) -> str:
        """
        Генерация отчета для модели регрессии
        
        Returns:
            Текстовый отчет
        """
        metrics = self.evaluation_results
        
        report = f"MODEL EVALUATION REPORT - {self.model_name}\n"
        report += "=" * 60 + "\n\n"
        report += "Task Type: Regression\n"
        report += f"Model Path: {self.model_path}\n\n"
        
        report += "REGRESSION METRICS\n"
        report += "-" * 60 + "\n"
        report += f"Mean Squared Error (MSE):      {metrics.get('mse', 'N/A'):.6f}\n"
        report += f"Root Mean Squared Error (RMSE): {metrics.get('rmse', 'N/A'):.6f}\n"
        report += f"Mean Absolute Error (MAE):     {metrics.get('mae', 'N/A'):.6f}\n"
        report += f"R-squared (R²):                {metrics.get('r2', 'N/A'):.6f}\n"
        report += f"Explained Variance:            {metrics.get('explained_variance', 'N/A'):.6f}\n"
        
        report += "\nINTERPRETATION\n"
        report += "-" * 60 + "\n"
        
        # Интерпретация R²
        r2 = metrics.get('r2', 0)
        if r2 > 0.9:
            report += "R² > 0.9: The model explains more than 90% of the variance in the data.\n"
            report += "This indicates an excellent fit.\n"
        elif r2 > 0.7:
            report += "0.7 < R² < 0.9: The model explains between 70% and 90% of the variance.\n"
            report += "This indicates a good fit.\n"
        elif r2 > 0.5:
            report += "0.5 < R² < 0.7: The model explains between 50% and 70% of the variance.\n"
            report += "This indicates a moderate fit.\n"
        elif r2 > 0.3:
            report += "0.3 < R² < 0.5: The model explains between 30% and 50% of the variance.\n"
            report += "This indicates a weak fit.\n"
        else:
            report += "R² < 0.3: The model explains less than 30% of the variance.\n"
            report += "This indicates a poor fit.\n"
        
        report += "\n" + "=" * 60 + "\n"
        report += f"End of report for {self.model_name}\n"
        
        return report


def main():
    """
    Пример использования оценщика моделей
    """
    # Создание демо-модели и данных для демонстрации
    from sklearn.datasets import make_classification, make_regression
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.model_selection import train_test_split
    import tempfile
    
    # Временный каталог для сохранения модели
    temp_dir = tempfile.TemporaryDirectory()
    
    # 1. Демонстрация для задачи классификации
    # Создаем синтетические данные
    X, y = make_classification(n_samples=1000, n_classes=3, n_features=20, 
                              n_informative=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Обучаем модель
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    # Сохраняем модель
    clf_path = os.path.join(temp_dir.name, "classification_model.pkl")
    joblib.dump(clf, clf_path)
    
    # Оцениваем модель
    evaluator = ModelEvaluator(clf_path, "RandomForestClassifier", "classification")
    metrics = evaluator.evaluate(X_test, y_test)
    
    # Выводим основные метрики
    logger.info(f"Classification model evaluation:")
    logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"F1 Score: {metrics['f1_score']:.4f}")
    
    # Визуализируем результаты
    evaluator.visualize_results(X_test, y_test, output_dir="model_evaluation/classification")
    
    # Генерируем отчет
    report = evaluator.generate_report("model_evaluation/classification/report.txt")
    
    # 2. Демонстрация для задачи регрессии
    # Создаем синтетические данные
    X_reg, y_reg = make_regression(n_samples=1000, n_features=20, 
                                 n_informative=10, noise=0.1, random_state=42)
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        X_reg, y_reg, test_size=0.3, random_state=42
    )
    
    # Обучаем модель
    reg = RandomForestRegressor(n_estimators=100, random_state=42)
    reg.fit(X_train_reg, y_train_reg)
    
    # Сохраняем модель
    reg_path = os.path.join(temp_dir.name, "regression_model.pkl")
    joblib.dump(reg, reg_path)
    
    # Оцениваем модель
    evaluator_reg = ModelEvaluator(reg_path, "RandomForestRegressor", "regression")
    metrics_reg = evaluator_reg.evaluate(X_test_reg, y_test_reg)
    
    # Выводим основные метрики
    logger.info(f"\nRegression model evaluation:")
    logger.info(f"RMSE: {metrics_reg['rmse']:.4f}")
    logger.info(f"R²: {metrics_reg['r2']:.4f}")
    
    # Визуализируем результаты
    evaluator_reg.visualize_results(X_test_reg, y_test_reg, output_dir="model_evaluation/regression")
    
    # Генерируем отчет
    report_reg = evaluator_reg.generate_report("model_evaluation/regression/report.txt")
    
    # Очистка временного каталога
    temp_dir.cleanup()


if __name__ == "__main__":
    main()
