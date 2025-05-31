#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import subprocess
import logging

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("dvc_commit")

def init_dvc(data_dir):
    """
    Инициализация DVC в проекте
    """
    if not os.path.exists(os.path.join(data_dir, '.dvc')):
        logger.info("Инициализация DVC...")
        subprocess.run(['dvc', 'init'], check=True)
        logger.info("DVC инициализирован успешно")
    else:
        logger.info("DVC уже инициализирован")

def add_data_to_dvc(data_path):
    """
    Добавление данных в отслеживание DVC
    """
    logger.info(f"Добавление данных в DVC: {data_path}")
    subprocess.run(['dvc', 'add', data_path], check=True)
    logger.info(f"Данные успешно добавлены в DVC: {data_path}")

def commit_changes(message):
    """
    Коммит изменений в Git
    """
    logger.info("Добавление .dvc файлов в Git")
    subprocess.run(['git', 'add', '*.dvc', '.gitignore'], check=True)
    
    logger.info(f"Коммит изменений: {message}")
    subprocess.run(['git', 'commit', '-m', message], check=True)
    
    logger.info("Изменения успешно закоммичены")

def push_data(remote=None):
    """
    Отправка данных в удаленное хранилище DVC
    """
    push_cmd = ['dvc', 'push']
    if remote:
        push_cmd.extend(['-r', remote])
    
    logger.info(f"Отправка данных в хранилище DVC: {remote if remote else 'default'}")
    subprocess.run(push_cmd, check=True)
    logger.info("Данные успешно отправлены")

def main():
    parser = argparse.ArgumentParser(description="Управление датасетом с помощью DVC")
    parser.add_argument('--data_dir', type=str, default='data', help='Путь к директории с данными')
    parser.add_argument('--message', type=str, default='Update dataset', help='Сообщение для коммита')
    parser.add_argument('--remote', type=str, help='Имя удаленного хранилища DVC')
    parser.add_argument('--push', action='store_true', help='Отправить данные в удаленное хранилище')
    
    args = parser.parse_args()
    
    # Проверка существования директории
    if not os.path.exists(args.data_dir):
        logger.error(f"Директория не существует: {args.data_dir}")
        return
    
    # Инициализация DVC
    init_dvc(args.data_dir)
    
    # Добавление данных
    add_data_to_dvc(args.data_dir)
    
    # Коммит изменений
    commit_changes(args.message)
    
    # Отправка данных, если указан флаг --push
    if args.push:
        push_data(args.remote)
    
    logger.info("Все операции выполнены успешно")

if __name__ == "__main__":
    main()
