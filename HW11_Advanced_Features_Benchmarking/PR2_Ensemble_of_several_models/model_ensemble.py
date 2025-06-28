# Updated version for PR
import torch
import numpy as np
from typing import List, Dict, Any, Union, Optional

class ModelEnsemble:
    """
    РљР»Р°СЃ, С‰Рѕ СЂРµР°Р»С–Р·СѓС” Р°РЅСЃР°РјР±Р»СЊ РјРѕРґРµР»РµР№ РјР°С€РёРЅРЅРѕРіРѕ РЅР°РІС‡Р°РЅРЅСЏ
    """
    def __init__(self, models: List[torch.nn.Module], weights: Optional[List[float]] = None, aggregation_method: str = 'weighted_average'):
        """
        Р†РЅС–С†С–Р°Р»С–Р·Р°С†С–СЏ Р°РЅСЃР°РјР±Р»СЋ РјРѕРґРµР»РµР№

        РџР°СЂР°РјРµС‚СЂРё:
        -----------
        models: СЃРїРёСЃРѕРє РјРѕРґРµР»РµР№ PyTorch РґР»СЏ Р°РЅСЃР°РјР±Р»СЋ
        weights: СЃРїРёСЃРѕРє РІР°РіРѕРІРёС… РєРѕРµС„С–С†С–С”РЅС‚С–РІ РґР»СЏ РєРѕР¶РЅРѕС— РјРѕРґРµР»С– (СЏРєС‰Рѕ None, РІСЃС– РјРѕРґРµР»С– РјР°СЋС‚СЊ РѕРґРЅР°РєРѕРІСѓ РІР°РіСѓ)
        aggregation_method: РјРµС‚РѕРґ Р°РіСЂРµРіР°С†С–С— СЂРµР·СѓР»СЊС‚Р°С‚С–РІ ('weighted_average', 'max_vote', 'softmax_average')
        """
        self.models = models
        self.num_models = len(models)

        # РџРµСЂРµРІС–СЂРєР° С‚Р° РЅРѕСЂРјР°Р»С–Р·Р°С†С–СЏ РІР°РіРѕРІРёС… РєРѕРµС„С–С†С–С”РЅС‚С–РІ
        if weights is None:
            self.weights = [1.0 / self.num_models] * self.num_models
        else:
            if len(weights) != self.num_models:
                raise ValueError("РљС–Р»СЊРєС–СЃС‚СЊ РІР°РіРѕРІРёС… РєРѕРµС„С–С†С–С”РЅС‚С–РІ РїРѕРІРёРЅРЅР° РґРѕСЂС–РІРЅСЋРІР°С‚Рё РєС–Р»СЊРєРѕСЃС‚С– РјРѕРґРµР»РµР№")

            # РќРѕСЂРјР°Р»С–Р·Р°С†С–СЏ РІР°РіРѕРІРёС… РєРѕРµС„С–С†С–С”РЅС‚С–РІ, С‰РѕР± С—С… СЃСѓРјР° РґРѕСЂС–РІРЅСЋРІР°Р»Р° 1
            sum_weights = sum(weights)
            self.weights = [w / sum_weights for w in weights]

        # РњРµС‚РѕРґ Р°РіСЂРµРіР°С†С–С—
        self.aggregation_method = aggregation_method
        self.valid_methods = ['weighted_average', 'max_vote', 'softmax_average']

        if self.aggregation_method not in self.valid_methods:
            raise ValueError(f"РќРµРїС–РґС‚СЂРёРјСѓРІР°РЅРёР№ РјРµС‚РѕРґ Р°РіСЂРµРіР°С†С–С—. Р”РѕСЃС‚СѓРїРЅС– РјРµС‚РѕРґРё: {', '.join(self.valid_methods)}")

        # РџРµСЂРµРІРµРґРµРЅРЅСЏ РјРѕРґРµР»РµР№ Сѓ СЂРµР¶РёРј РѕС†С–РЅРєРё
        for model in self.models:
            model.eval()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Р’РёРєРѕРЅР°РЅРЅСЏ РїСЂРѕРіРЅРѕР·СѓРІР°РЅРЅСЏ Р°РЅСЃР°РјР±Р»РµРј РјРѕРґРµР»РµР№

        РџР°СЂР°РјРµС‚СЂРё:
        -----------
        x: РІС…С–РґРЅРёР№ С‚РµРЅР·РѕСЂ РґР»СЏ РїСЂРѕРіРЅРѕР·СѓРІР°РЅРЅСЏ

        РџРѕРІРµСЂС‚Р°С”:
        -----------
        Р°РіСЂРµРіРѕРІР°РЅРёР№ СЂРµР·СѓР»СЊС‚Р°С‚ РїСЂРѕРіРЅРѕР·СѓРІР°РЅРЅСЏ
        """
        return self.predict(x)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Р’РёРєРѕРЅР°РЅРЅСЏ РїСЂРѕРіРЅРѕР·СѓРІР°РЅРЅСЏ Р°РЅСЃР°РјР±Р»РµРј РјРѕРґРµР»РµР№

        РџР°СЂР°РјРµС‚СЂРё:
        -----------
        x: РІС…С–РґРЅРёР№ С‚РµРЅР·РѕСЂ РґР»СЏ РїСЂРѕРіРЅРѕР·СѓРІР°РЅРЅСЏ

        РџРѕРІРµСЂС‚Р°С”:
        -----------
        Р°РіСЂРµРіРѕРІР°РЅРёР№ СЂРµР·СѓР»СЊС‚Р°С‚ РїСЂРѕРіРЅРѕР·СѓРІР°РЅРЅСЏ
        """
        predictions = []

        # РћС‚СЂРёРјР°РЅРЅСЏ РїСЂРѕРіРЅРѕР·С–РІ РІС–Рґ РєРѕР¶РЅРѕС— РјРѕРґРµР»С–
        with torch.no_grad():
            for model in self.models:
                output = model(x)
                predictions.append(output)

        # РђРіСЂРµРіР°С†С–СЏ СЂРµР·СѓР»СЊС‚Р°С‚С–РІ
        if self.aggregation_method == 'weighted_average':
            return self._weighted_average(predictions)
        elif self.aggregation_method == 'max_vote':
            return self._max_vote(predictions)
        elif self.aggregation_method == 'softmax_average':
            return self._softmax_average(predictions)

        # Р—Р° Р·Р°РјРѕРІС‡СѓРІР°РЅРЅСЏРј РІРёРєРѕСЂРёСЃС‚РѕРІСѓС”РјРѕ Р·РІР°Р¶РµРЅРµ СЃРµСЂРµРґРЅС”
        return self._weighted_average(predictions)

    def _weighted_average(self, predictions: List[torch.Tensor]) -> torch.Tensor:
        """
        РђРіСЂРµРіР°С†С–СЏ Р·Р° РјРµС‚РѕРґРѕРј Р·РІР°Р¶РµРЅРѕРіРѕ СЃРµСЂРµРґРЅСЊРѕРіРѕ

        РџР°СЂР°РјРµС‚СЂРё:
        -----------
        predictions: СЃРїРёСЃРѕРє РїСЂРѕРіРЅРѕР·С–РІ РІС–Рґ РєРѕР¶РЅРѕС— РјРѕРґРµР»С–

        РџРѕРІРµСЂС‚Р°С”:
        -----------
        Р·РІР°Р¶РµРЅРµ СЃРµСЂРµРґРЅС” РїСЂРѕРіРЅРѕР·С–РІ
        """
        weighted_preds = [pred * weight for pred, weight in zip(predictions, self.weights)]
        return sum(weighted_preds)

    def _max_vote(self, predictions: List[torch.Tensor]) -> torch.Tensor:
        """
        РђРіСЂРµРіР°С†С–СЏ Р·Р° РјРµС‚РѕРґРѕРј РјР°РєСЃРёРјР°Р»СЊРЅРѕРіРѕ РіРѕР»РѕСЃСѓРІР°РЅРЅСЏ

        РџР°СЂР°РјРµС‚СЂРё:
        -----------
        predictions: СЃРїРёСЃРѕРє РїСЂРѕРіРЅРѕР·С–РІ РІС–Рґ РєРѕР¶РЅРѕС— РјРѕРґРµР»С–

        РџРѕРІРµСЂС‚Р°С”:
        -----------
        СЂРµР·СѓР»СЊС‚Р°С‚ РјР°РєСЃРёРјР°Р»СЊРЅРѕРіРѕ РіРѕР»РѕСЃСѓРІР°РЅРЅСЏ
        """
        # Р”Р»СЏ РєР»Р°СЃРёС„С–РєР°С†С–С— - РѕР±РёСЂР°С”РјРѕ РєР»Р°СЃ Р· РјР°РєСЃРёРјР°Р»СЊРЅРёРј РіРѕР»РѕСЃСѓРІР°РЅРЅСЏРј
        if predictions[0].dim() > 1 and predictions[0].size(1) > 1:  # РЇРєС‰Рѕ С†Рµ Р·Р°РґР°С‡Р° РєР»Р°СЃРёС„С–РєР°С†С–С—
            # РљРѕРЅРІРµСЂС‚Р°С†С–СЏ Р»РѕРіС–С‚С–РІ Сѓ РєР»Р°СЃРё
            class_predictions = [torch.argmax(pred, dim=1) for pred in predictions]

            # РџС–РґСЂР°С…СѓРЅРѕРє РіРѕР»РѕСЃС–РІ РґР»СЏ РєРѕР¶РЅРѕРіРѕ РєР»Р°СЃСѓ
            batch_size = predictions[0].size(0)
            num_classes = predictions[0].size(1)

            result = torch.zeros((batch_size, num_classes), device=predictions[0].device)

            for i, class_pred in enumerate(class_predictions):
                for b in range(batch_size):
                    result[b, class_pred[b]] += self.weights[i]

            return result
        else:  # Р”Р»СЏ СЂРµРіСЂРµСЃС–С— Р°Р±Рѕ С–РЅС€РёС… Р·Р°РІРґР°РЅСЊ
            # РћР±РёСЂР°С”РјРѕ РїСЂРѕРіРЅРѕР· Р· РјР°РєСЃРёРјР°Р»СЊРЅРѕСЋ РІР°РіРѕСЋ
            weighted_preds = [(pred * weight) for pred, weight in zip(predictions, self.weights)]
            return torch.stack(weighted_preds).max(dim=0)[0]

    def _softmax_average(self, predictions: List[torch.Tensor]) -> torch.Tensor:
        """
        РђРіСЂРµРіР°С†С–СЏ Р·Р° РјРµС‚РѕРґРѕРј СѓСЃРµСЂРµРґРЅРµРЅРЅСЏ softmax-Р№РјРѕРІС–СЂРЅРѕСЃС‚РµР№

        РџР°СЂР°РјРµС‚СЂРё:
        -----------
        predictions: СЃРїРёСЃРѕРє РїСЂРѕРіРЅРѕР·С–РІ РІС–Рґ РєРѕР¶РЅРѕС— РјРѕРґРµР»С–

        РџРѕРІРµСЂС‚Р°С”:
        -----------
        СѓСЃРµСЂРµРґРЅРµРЅС– softmax-Р№РјРѕРІС–СЂРЅРѕСЃС‚С–
        """
        softmax = torch.nn.Softmax(dim=1)

        # РџРµСЂРµС‚РІРѕСЂРµРЅРЅСЏ Р»РѕРіС–С‚С–РІ Сѓ softmax-Р№РјРѕРІС–СЂРЅРѕСЃС‚С–
        softmax_preds = [softmax(pred) for pred in predictions]

        # Р—РІР°Р¶РµРЅРµ СѓСЃРµСЂРµРґРЅРµРЅРЅСЏ Р№РјРѕРІС–СЂРЅРѕСЃС‚РµР№
        weighted_softmax = [pred * weight for pred, weight in zip(softmax_preds, self.weights)]
        return sum(weighted_softmax)

    def get_individual_predictions(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        РћС‚СЂРёРјР°РЅРЅСЏ С–РЅРґРёРІС–РґСѓР°Р»СЊРЅРёС… РїСЂРѕРіРЅРѕР·С–РІ РІС–Рґ РєРѕР¶РЅРѕС— РјРѕРґРµР»С–

        РџР°СЂР°РјРµС‚СЂРё:
        -----------
        x: РІС…С–РґРЅРёР№ С‚РµРЅР·РѕСЂ РґР»СЏ РїСЂРѕРіРЅРѕР·СѓРІР°РЅРЅСЏ

        РџРѕРІРµСЂС‚Р°С”:
        -----------
        СЃРїРёСЃРѕРє РїСЂРѕРіРЅРѕР·С–РІ РІС–Рґ РєРѕР¶РЅРѕС— РјРѕРґРµР»С–
        """
        predictions = []

        with torch.no_grad():
            for model in self.models:
                output = model(x)
                predictions.append(output)

        return predictions

