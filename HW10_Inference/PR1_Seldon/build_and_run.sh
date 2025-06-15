#!/bin/bash

# Збірка Docker-образу
docker build -t resnet50-classifier:latest .

# Запуск контейнера
docker run -p 9000:9000 resnet50-classifier:latest
