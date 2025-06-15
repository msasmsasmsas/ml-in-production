@echo off
echo Запуск FastAPI сервера для ResNet50
@echo off
echo Запуск FastAPI сервера для ResNet50

:: Активація віртуального середовища
call ..\..\HW2-Infrastructure-setup\venv\Scripts\activate.bat

:: Встановлення необхідних пакетів якщо вони не встановлені
pip install fastapi uvicorn python-multipart

:: Запуск сервера
python simple_server.py

pause
:: Активація віртуального середовища
call ..\..\HW2-Infrastructure-setup\venv\Scripts\activate.bat

:: Встановлення необхідних пакетів якщо вони не встановлені
pip install fastapi uvicorn python-multipart

:: Запуск сервера
python simple_server.py

pause
