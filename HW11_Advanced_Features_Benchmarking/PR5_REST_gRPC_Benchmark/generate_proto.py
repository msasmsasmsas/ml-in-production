#!/usr/bin/env python
"""
Скрипт для генерації Python-коду з .proto файлів
"""

import os
import sys
import subprocess
from pathlib import Path

def generate_grpc_code(proto_file, output_dir="."):
    """
    Генерує Python-код з .proto файлу за допомогою protoc

    Параметри:
    -----------
    proto_file: шлях до .proto файлу
    output_dir: директорія для вихідних файлів
    """
    proto_file = Path(proto_file)

    if not proto_file.exists():
        print(f"Помилка: файл {proto_file} не існує")
        return False

    try:
        # Команда для генерації Python-коду
        cmd = [
            "python", "-m", "grpc_tools.protoc",
            f"--proto_path={proto_file.parent}",
            f"--python_out={output_dir}",
            f"--grpc_python_out={output_dir}",
            str(proto_file)
        ]

        print(f"Виконання команди: {' '.join(cmd)}")
        subprocess.check_call(cmd)

        print(f"Успішно згенеровано Python-код з {proto_file}")
        return True

    except subprocess.CalledProcessError as e:
        print(f"Помилка при генерації коду: {e}")
        return False
    except Exception as e:
        print(f"Неочікувана помилка: {e}")
        return False

if __name__ == "__main__":
    # Шлях до .proto файлу за замовчуванням
    default_proto = "proto/inference.proto"

    # Отримання шляху з аргументів командного рядка, якщо вказано
    proto_file = sys.argv[1] if len(sys.argv) > 1 else default_proto

    success = generate_grpc_code(proto_file)

    if not success:
        sys.exit(1)
