import torch
import numpy as np
from typing import List, Dict, Any, Union, Optional

class ModelEnsemble:
    """
    Клас, що реалізує ансамбль моделей машинного навчання
    """
    def __init__(self, models: List[torch.nn.Module], weights: Optional[List[float]] = None, aggregation_method: str = 'weighted_average'):
        """
        Ініціалізація ансамблю моделей

        Параметри:
        -----------
        models: список моделей PyTorch для ансамблю
        weights: список вагових коефіцієнтів для кожної моделі (якщо None, всі моделі мають однакову вагу)
        aggregation_method: метод агрегації результатів ('weighted_average', 'max_vote', 'softmax_average')
        """
        self.models = models
        self.num_models = len(models)

        # Перевірка та нормалізація вагових коефіцієнтів
        if weights is None:
            self.weights = [1.0 / self.num_models] * self.num_models
        else:
            if len(weights) != self.num_models:
                raise ValueError("Кількість вагових коефіцієнтів повинна дорівнювати кількості моделей")

            # Нормалізація вагових коефіцієнтів, щоб їх сума дорівнювала 1
            sum_weights = sum(weights)
            self.weights = [w / sum_weights for w in weights]

        # Метод агрегації
        self.aggregation_method = aggregation_method
        self.valid_methods = ['weighted_average', 'max_vote', 'softmax_average']

        if self.aggregation_method not in self.valid_methods:
            raise ValueError(f"Непідтримуваний метод агрегації. Доступні методи: {', '.join(self.valid_methods)}")

        # Переведення моделей у режим оцінки
        for model in self.models:
            model.eval()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Виконання прогнозування ансамблем моделей

        Параметри:
        -----------
        x: вхідний тензор для прогнозування

        Повертає:
        -----------
        агрегований результат прогнозування
        """
        return self.predict(x)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Виконання прогнозування ансамблем моделей

        Параметри:
        -----------
        x: вхідний тензор для прогнозування

        Повертає:
        -----------
        агрегований результат прогнозування
        """
        predictions = []

        # Отримання прогнозів від кожної моделі
        with torch.no_grad():
            for model in self.models:
                output = model(x)
                predictions.append(output)

        # Агрегація результатів
        if self.aggregation_method == 'weighted_average':
            return self._weighted_average(predictions)
        elif self.aggregation_method == 'max_vote':
            return self._max_vote(predictions)
        elif self.aggregation_method == 'softmax_average':
            return self._softmax_average(predictions)

        # За замовчуванням використовуємо зважене середнє
        return self._weighted_average(predictions)

    def _weighted_average(self, predictions: List[torch.Tensor]) -> torch.Tensor:
        """
        Агрегація за методом зваженого середнього

        Параметри:
        -----------
        predictions: список прогнозів від кожної моделі

        Повертає:
        -----------
        зважене середнє прогнозів
        """
        weighted_preds = [pred * weight for pred, weight in zip(predictions, self.weights)]
        return sum(weighted_preds)

    def _max_vote(self, predictions: List[torch.Tensor]) -> torch.Tensor:
        """
        Агрегація за методом максимального голосування

        Параметри:
        -----------
        predictions: список прогнозів від кожної моделі

        Повертає:
        -----------
        результат максимального голосування
        """
        # Для класифікації - обираємо клас з максимальним голосуванням
        if predictions[0].dim() > 1 and predictions[0].size(1) > 1:  # Якщо це задача класифікації
            # Конвертація логітів у класи
            class_predictions = [torch.argmax(pred, dim=1) for pred in predictions]

            # Підрахунок голосів для кожного класу
            batch_size = predictions[0].size(0)
            num_classes = predictions[0].size(1)

            result = torch.zeros((batch_size, num_classes), device=predictions[0].device)

            for i, class_pred in enumerate(class_predictions):
                for b in range(batch_size):
                    result[b, class_pred[b]] += self.weights[i]

            return result
        else:  # Для регресії або інших завдань
            # Обираємо прогноз з максимальною вагою
            weighted_preds = [(pred * weight) for pred, weight in zip(predictions, self.weights)]
            return torch.stack(weighted_preds).max(dim=0)[0]

    def _softmax_average(self, predictions: List[torch.Tensor]) -> torch.Tensor:
        """
        Агрегація за методом усереднення softmax-ймовірностей

        Параметри:
        -----------
        predictions: список прогнозів від кожної моделі

        Повертає:
        -----------
        усереднені softmax-ймовірності
        """
        softmax = torch.nn.Softmax(dim=1)

        # Перетворення логітів у softmax-ймовірності
        softmax_preds = [softmax(pred) for pred in predictions]

        # Зважене усереднення ймовірностей
        weighted_softmax = [pred * weight for pred, weight in zip(softmax_preds, self.weights)]
        return sum(weighted_softmax)

    def get_individual_predictions(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Отримання індивідуальних прогнозів від кожної моделі

        Параметри:
        -----------
        x: вхідний тензор для прогнозування

        Повертає:
        -----------
        список прогнозів від кожної моделі
        """
        predictions = []

        with torch.no_grad():
            for model in self.models:
                output = model(x)
                predictions.append(output)

        return predictions
