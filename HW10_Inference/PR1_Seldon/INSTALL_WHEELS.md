# Установка зависимостей через wheel-файлы

Если у вас проблемы с компиляцией пакетов в Windows, используйте готовые wheel-файлы.

## Шаг 1: Скачайте wheel-файлы для вашей версии Python

1. Перейдите на [Unofficial Windows Binaries for Python](https://www.lfd.uci.edu/~gohlke/pythonlibs/)
2. Скачайте следующие файлы для вашей версии Python (например, для Python 3.12 и Windows 64-bit):
   - PyYAML‑6.0.1‑cp312‑cp312‑win_amd64.whl
   - Другие проблемные пакеты при необходимости

## Шаг 2: Установите wheel-файлы

```bash
# Активируйте виртуальное окружение
.\venv\Scripts\activate

# Установите wheel-файлы (замените путь на фактический)
pip install C:/path/to/PyYAML‑6.0.1‑cp312‑cp312‑win_amd64.whl

# После этого установите остальные зависимости
pip install numpy pandas torch torchvision pillow requests pytest seldon-core==1.17.0
```

## Шаг 3: Используйте Docker вместо локальной установки

Для полного обхода проблем с пакетами, используйте Docker:

```bash
# Загрузите образ Seldon Core
docker pull seldonio/seldon-core:1.17.0
```

Дальше следуйте инструкциям по Docker в README.md.