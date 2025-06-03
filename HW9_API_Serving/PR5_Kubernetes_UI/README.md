# Конфігурація Kubernetes для розгортання користувацьких інтерфейсів (UI)

## Опіс проєкту

Цей проєкт містить конфігураційні файли Kubernetes для розгортання веб-інтерфейсів, які взаємодіють з API моделі розпізнавання загроз сільськогосподарським культурам. Включено конфігурації для розгортання двох інтерфейсів: Streamlit та Gradio.

## Компоненти Kubernetes

### Deployments

#### Streamlit UI Deployment
- 2 репліки для забезпечення доступності
- Стратегія оновлення Rolling Update з нульовим простоєм
- Контейнер на базі Python з веб-інтерфейсом Streamlit
- Проби готовності та живучості для моніторингу стану

#### Gradio UI Deployment
- 2 репліки для забезпечення доступності
- Аналогічна стратегія оновлення
- Контейнер з веб-інтерфейсом Gradio
- Проби для перевірки стану сервісу

### Services

Для кожного інтерфейсу створено окремий сервіс типу ClusterIP:
- Streamlit UI Service: перенаправлення з порту 80 на порт 8501 додатка
- Gradio UI Service: перенаправлення з порту 80 на порт 7860 додатка

### Ingress

Налаштовує зовнішній доступ до обох інтерфейсів:
- Доменне ім'я для Streamlit: streamlit.crop-threat-detection.example.com
- Доменне ім'я для Gradio: gradio.crop-threat-detection.example.com
- Перенаправлення на відповідні сервіси

### ConfigMap

Централізоване зберігання конфігурацій для обох інтерфейсів:
- Порти для Streamlit та Gradio
- Адреси серверів
- Налаштування таймаутів для API

## Dockerfile-и для UI

### Dockerfile-Streamlit
- Базується на офіційному образі Python
- Встановлює необхідні залежності з requirements.txt
- Копіює код додатка Streamlit
- Налаштовує сервер Streamlit на порт 8501

### Dockerfile-Gradio
- Аналогічна базова конфігурація
- Встановлює залежності Gradio
- Налаштовує запуск сервера на порт 7860
- Включає папку для прикладів зображень

## Інструкція з розгортання

1. Переконайтеся, що namespace ml-serving існує:
   ```
   kubectl create namespace ml-serving
   ```

2. Створіть ConfigMap з конфігурацією:
   ```
   kubectl apply -f deployment.yaml
   ```

3. Створіть образи Docker для інтерфейсів:
   ```
   docker build -f Dockerfile-Streamlit -t crop-threat-streamlit-ui:latest .
   docker build -f Dockerfile-Gradio -t crop-threat-gradio-ui:latest .
   ```

4. Застосуйте конфігураційні файли:
   ```
   kubectl apply -f deployment.yaml
   ```

5. Перевірте статус розгортання:
   ```
   kubectl get pods -n ml-serving
   kubectl get services -n ml-serving
   kubectl get ingress -n ml-serving
   ```

## Налаштування DNS

Для доступу до інтерфейсів через доменні імена необхідно налаштувати DNS або додати записи до файлу hosts:
