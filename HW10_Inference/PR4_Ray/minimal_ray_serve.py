#!/usr/bin/env python3

"""
Мінімальний приклад Ray Serve для демонстрації.
"""

import ray
from ray import serve
import time

# Ініціалізація Ray, якщо ще не ініціалізовано
if not ray.is_initialized():
    ray.init()

# Запуск Ray Serve
serve.start()

# Простий обробник запитів
@serve.deployment(route_prefix="/")
class SimpleDeployment:
    async def __call__(self, request):
        # Обробка запиту залежно від шляху
        if request.method == "GET":
            if request.url.path == "/health":
                return {"status": "healthy"}
            elif request.url.path == "/":
                return {"message": "Привіт від Ray Serve!"}
            else:
                return {"error": "Endpoint not found"}
        return {"error": "Method not allowed"}

# Розгортання додатку
deployment = SimpleDeployment.bind()
serve.run(deployment)

print("\nМінімальний Ray Serve додаток запущено!")
print("API доступний за адресою: http://localhost:8000/")
print("\nДоступні ендпоінти:")
print("  GET  /        - Привітання")
print("  GET  /health  - Перевірка стану")

# Тримаємо процес запущеним
try:
    while True:
        time.sleep(60)
except KeyboardInterrupt:
    print("\nЗавершення роботи...")
