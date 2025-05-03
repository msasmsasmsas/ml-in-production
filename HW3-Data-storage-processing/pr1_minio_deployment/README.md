Розгортання MinIO: Локально, Docker, Kubernetes
Цей документ містить інструкції для розгортання MinIO у трьох режимах: локально, через Docker та на Kubernetes.
Локальне розгортання
Вимоги

Python 3.12
MinIO сервер (завантажте з https://min.io/download)
ОС: WSL (Ubuntu)

Кроки

Завантажте MinIO для Linux:
```
wget https://dl.min.io/server/minio/release/linux-amd64/minio
chmod +x minio
```

Створіть папку для даних:
```
mkdir ~/minio-data
```

Запустіть MinIO сервер:
```
./minio server ~/minio-data --console-address ":9001"
```

Відкрийте браузер за адресою 
http://localhost:9001 
та увійдіть (логін: minioadmin, пароль: minioadmin).

Розгортання через Docker
Вимоги

Docker (встановіть у WSL: sudo apt install docker.io)

Кроки

Завантажте образ MinIO:
```
docker pull minio/minio
```



Запустіть контейнер:
```
docker run -p 9000:9000 -p 9001:9001 --name minio -v minio-data:/data -e "MINIO_ROOT_USER=admin" -e "MINIO_ROOT_PASSWORD=password" minio/minio server /data --console-address ":9001"
```

Перейдіть до http://localhost:9001 для доступу до консолі.

Розгортання на Kubernetes
Вимоги

Minikube (встановіть у WSL: https://minikube.sigs.k8s.io/docs/start/)
kubectl (встановіть: sudo apt install kubectl)

Кроки

Запустіть Minikube:
```
minikube start
```

Створіть файл minio-deployment.yaml:
```
apiVersion: apps/v1
kind: Deployment
metadata:
  name: minio
spec:
  replicas: 1
  selector:
    matchLabels:
      app: minio
  template:
    metadata:
      labels:
        app: minio
    spec:
      containers:
      - name: minio
        image: minio/minio
        args:
        - server
        - /data
        - --console-address
        - ":9001"
        env:
        - name: MINIO_ROOT_USER
          value: "admin"
        - name: MINIO_ROOT_PASSWORD
          value: "password"
        ports:
        - containerPort: 9000
        - containerPort: 9001
        volumeMounts:
        - name: data
          mountPath: /data
      volumes:
      - name: data
        emptyDir: {}
---
apiVersion: v1
kind: Service
metadata:
  name: minio-service
spec:
  ports:
  - port: 9000
    targetPort: 9000
    name: minio
  - port: 9001
    targetPort: 9001
    name: console
  selector:
    app: minio
  type: NodePort
```

Застосуйте конфігурацію:
```
kubectl apply -f minio-deployment.yaml
```

Отримайте URL для доступу:
```
minikube service minio-service --url
```

Перейдіть за отриманим URL для доступу до консолі MinIO.

