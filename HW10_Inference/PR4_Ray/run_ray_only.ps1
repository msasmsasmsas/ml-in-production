# Скрипт для запуску тільки Ray Serve

# Встановлюємо змінну середовища PYTHONPATH
$env:PYTHONPATH = "E:\ml-in-production"

# Запускаємо Ray Serve без uvicorn
Write-Host "Запуск Ray Serve без uvicorn..."
python -m app.main --serve-only
