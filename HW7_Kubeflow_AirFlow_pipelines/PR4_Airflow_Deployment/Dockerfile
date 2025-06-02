FROM apache/airflow:2.7.1

USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

USER airflow

# Установка необходимых Python-библиотек
RUN pip install --no-cache-dir \
    scikit-learn \
    numpy \
    pandas \
    matplotlib \
    torch \
    torchvision