# Простий скрипт для запуску та тестування Ray Serve

# Додаємо модуль до PYTHONPATH якщо потрібно
$env:PYTHONPATH = "E:\ml-in-production"

# Перед запуском, зупиняємо попередні екземпляри Ray
Write-Host "Зупинка попередніх екземплярів Ray..."
python -c "import ray; ray.shutdown() if ray.is_initialized() else None"

# Запуск додатку тільки з Ray Serve
Write-Host "Запуск Ray Serve додатку..."
python -m app.main --serve-only
