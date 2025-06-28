#!/usr/bin/env python3

"""
Main entry point for Ray Serve application.
"""

import ray
from ray import serve
from fastapi import FastAPI
import argparse
import sys
import os

# Умовний імпорт uvicorn (може не знадобитися при окремому запуску serve.run)
try:
    import uvicorn
except ImportError:
    uvicorn = None

# Додаємо корінь проекту до sys.path для правильного імпорту модулів
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Тепер імпортуємо модулі з правильними шляхами
from app.deployments.classifier import ImageClassifierDeployment
from app.deployments.router import RouterDeployment

def create_app():
    """Create and configure the Ray Serve application."""
    # Initialize Ray and Ray Serve
    if not ray.is_initialized():
        ray.init()
    serve.start(detached=True)

    # Create a FastAPI app as the ingress
    app = FastAPI(title="Ray Serve ML API",
                 description="API for ML model serving with Ray Serve",
                 version="1.0.0")

    # Deploy the image classifier
    classifier = serve.deployment(
        name="image_classifier",
        ray_actor_options={"num_cpus": 1, "num_gpus": 0},
        # Ray Serve в нових версіях не підтримує max_concurrent_queries
        # max_concurrent_queries=10,
        autoscaling_config={"min_replicas": 1, "max_replicas": 3}
    )(ImageClassifierDeployment).bind()

    # Deploy the router that handles the HTTP interface
    router = serve.deployment(
        name="router",
        ray_actor_options={"num_cpus": 1},
        # Ray Serve в нових версіях не підтримує max_concurrent_queries
        # max_concurrent_queries=20
    )(RouterDeployment).bind(classifier)

    # Deploy both components - передаємо route_prefix тут
    serve.run(router, route_prefix="/")

    return app

def main():
    parser = argparse.ArgumentParser(description="Ray Serve ML API")
    parser.add_argument(
        "--host", type=str, default="127.0.0.1", help="Host to run the server on"
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="Port to run the server on"
    )
    parser.add_argument(
        "--reload", action="store_true", help="Enable auto-reload for development"
    )
    parser.add_argument(
        "--serve-only", action="store_true", help="Run only Ray Serve without uvicorn"
    )
    parser.add_argument(
        "--no-uvicorn", action="store_true", help="Skip uvicorn and just run Ray Serve"
    )
    args = parser.parse_args()

    # Create the app
    app = create_app()

    # If uvicorn is available and not explicitly disabled, run the FastAPI server
    if uvicorn is not None and not args.no_uvicorn:
        uvicorn.run(
            "app.main:app",  # Повний шлях до модуля
            host=args.host,
            port=args.port,
            reload=args.reload
        )
    else:
        print("\nRay Serve запущено успішно без uvicorn!")
        print("API доступний за адресою: http://localhost:8000/")
        print("\nДоступні ендпоінти:")
        print("  GET  /health    - Перевірка стану сервера")
        print("  GET  /metadata  - Отримання метаданих моделі")
        print("  POST /predict   - Класифікація зображення")

        # Простий блокуючий виклик, щоб утримати процес
        try:
            while True:
                import time
                time.sleep(60)
        except KeyboardInterrupt:
            print("\nЗавершення роботи сервера...")

if __name__ == "__main__":
    main()
