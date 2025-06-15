@echo off
echo Тестирование API для ResNet50
@echo off
echo Тестування API для ResNet50

:: Активація віртуального середовища
call ..\..\HW2-Infrastructure-setup\venv\Scripts\activate.bat

:: Запуск тесту
python test_api.py

pause
:: Активация виртуального окружения
call ..\..\HW2-Infrastructure-setup\venv\Scripts\activate.bat

:: Запуск теста
python test_api.py

pause
