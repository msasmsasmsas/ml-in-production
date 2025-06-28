#!/usr/bin/env python3
"""
Простой сервер FastAPI для модели ResNet50 (альтернатива Seldon Core).
"""

import logging
import os
import io
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Инициализация FastAPI
app = FastAPI(title="ResNet50 Image Classifier API")

# Добавление CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Импортируем модель
from model.ResNet50Classifier import ResNet50Classifier

# Глобальный экземпляр модели
model = None
#!/usr/bin/env python3
"""
Простий сервер FastAPI для моделі ResNet50 (альтернатива Seldon Core).
"""

import logging
import os
import io
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware

# Налаштування логування
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ініціалізація FastAPI
app = FastAPI(title="API класифікатора зображень ResNet50")

# Додавання CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Імпортуємо модель
from model.ResNet50Classifier import ResNet50Classifier

# Глобальний екземпляр моделі
model = None

@app.on_event("startup")
async def startup_event():
    """Ініціалізація моделі при запуску сервера"""
    global model
    logger.info("Ініціалізація моделі ResNet50Classifier")
    model = ResNet50Classifier()
    logger.info("Модель успішно ініціалізована")

@app.get("/")
async def root():
    """Кореневий маршрут для перевірки працездатності сервера"""
    return {"message": "API класифікатора зображень ResNet50 працює", "model": "ResNet50"}

@app.post("/api/v1.0/predictions")
async def predict_seldon_compatible(request: dict):
    """Сумісний з Seldon Core API маршрут для прогнозування"""
    global model

    try:
        # Витягнення даних із запиту у форматі Seldon Core
        if "data" in request and "ndarray" in request["data"]:
            # Формат Seldon Core: {"data": {"ndarray": [...]}}
            image_array = np.array(request["data"]["ndarray"], dtype=np.uint8)

            # Виконання прогнозування
            result = model.predict(image_array)

            # Повертаємо результат у форматі Seldon Core
            return {"data": result}
        else:
            return {"error": "Невірний формат запиту. Очікується {\"data\": {\"ndarray\": [...]}}"}

    except Exception as e:
        logger.error(f"Помилка при обробці запиту: {str(e)}")
        return {"error": str(e)}

@app.post("/predict")
async def predict_image(file: UploadFile = File(...), confidence: float = Form(0.5)):
    """Простий API для прогнозування на основі завантаженого зображення"""
    global model

    try:
        # Читання зображення
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Виконання прогнозування
        result = model.predict(image)

        # Фільтрація результатів за порогом впевненості
        if "predictions" in result:
            result["predictions"] = [
                p for p in result["predictions"] if p["probability"] >= confidence
            ]

        return result

    except Exception as e:
        logger.error(f"Помилка при обробці зображення: {str(e)}")
        return {"error": str(e)}

# Запуск сервера
if __name__ == "__main__":
    import uvicorn
    # Визначимо порт із змінної середовища або використаємо значення за замовчуванням
    port = int(os.environ.get("PORT", 9000))
    uvicorn.run("simple_server:app", host="0.0.0.0", port=port, reload=False)
@app.on_event("startup")
async def startup_event():
    """Инициализация модели при запуске сервера"""
    global model
    logger.info("Инициализация модели ResNet50Classifier")
    model = ResNet50Classifier()
    logger.info("Модель успешно инициализирована")

@app.get("/")
async def root():
    """Корневой маршрут для проверки работоспособности сервера"""
    return {"message": "ResNet50 Image Classifier API работает", "model": "ResNet50"}

@app.post("/api/v1.0/predictions")
async def predict_seldon_compatible(request: dict):
    """Совместимый с Seldon Core API маршрут для предсказаний"""
    global model

    try:
        # Извлечение данных из запроса в формате Seldon Core
        if "data" in request and "ndarray" in request["data"]:
            # Формат Seldon Core: {"data": {"ndarray": [...]}}
            image_array = np.array(request["data"]["ndarray"], dtype=np.uint8)

            # Выполнение предсказания
            result = model.predict(image_array)

            # Возвращаем результат в формате Seldon Core
            return {"data": result}
        else:
            return {"error": "Неверный формат запроса. Ожидается {\"data\": {\"ndarray\": [...]}}"}

    except Exception as e:
        logger.error(f"Ошибка при обработке запроса: {str(e)}")
        return {"error": str(e)}

@app.post("/predict")
async def predict_image(file: UploadFile = File(...), confidence: float = Form(0.5)):
    """Простой API для предсказания на основе загруженного изображения"""
    global model

    try:
        # Чтение изображения
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Выполнение предсказания
        result = model.predict(image)

        # Фильтрация результатов по порогу уверенности
        if "predictions" in result:
            result["predictions"] = [
                p for p in result["predictions"] if p["probability"] >= confidence
            ]

        return result

    except Exception as e:
        logger.error(f"Ошибка при обработке изображения: {str(e)}")
        return {"error": str(e)}

# Запуск сервера
if __name__ == "__main__":
    import uvicorn
    # Определим порт из переменной окружения или используем значение по умолчанию
    port = int(os.environ.get("PORT", 9000))
    uvicorn.run("simple_server:app", host="0.0.0.0", port=port, reload=False)
