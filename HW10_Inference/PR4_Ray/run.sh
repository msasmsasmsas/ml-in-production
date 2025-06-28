#!/bin/bash

# Скрипт для запуску Ray Serve додатку

# Додавання кореневого каталогу до PYTHONPATH
export PYTHONPATH="$PWD:$PYTHONPATH"

# Запуск додатку
echo "Запуск Ray Serve додатку..."
python -m app.main
