# Скрипт для запуску Ray Serve додатку

# Додавання кореневого каталогу до PYTHONPATH
$env:PYTHONPATH = "E:\ml-in-production"

# Запуск додатку
Write-Host "Запуск Ray Serve додатку..."
python -m app.main
