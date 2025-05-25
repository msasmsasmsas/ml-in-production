#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import wandb

def main():
    parser = argparse.ArgumentParser(description="Логин в Weights & Biases")
    parser.add_argument("--key", required=True, help="API ключ для wandb")
    args = parser.parse_args()
    
    # Авторизация в wandb
    wandb.login(key=args.key)
    print("Успешный вход в wandb")
    
    # Сохранение ключа в переменную окружения (для текущей сессии)
    os.environ["WANDB_API_KEY"] = args.key
    print("API ключ установлен в переменную окружения WANDB_API_KEY")
    
    # Инструкции для пользователя
    print("\nДля постоянного использования этого ключа:")
    print("1. В Windows: setx WANDB_API_KEY", args.key)
    print("2. В Linux/MacOS: export WANDB_API_KEY=", args.key, "&& echo 'export WANDB_API_KEY=", 
          args.key, "' >> ~/.bashrc")
    
if __name__ == "__main__":
    main()
