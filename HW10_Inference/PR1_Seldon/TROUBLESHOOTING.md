# Руководство по устранению проблем для PR1_Seldon

## Проблема с длинными путями в Windows

При установке `seldon-core-microservice` в Windows часто возникает ошибка:

```
error: unable to create file ... Filename too long
fatal: unable to checkout working tree
```

### Решение 1: Включение поддержки длинных путей

1. Откройте PowerShell с правами администратора и выполните:

```powershell
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
```

2. Настройте Git для поддержки длинных путей:

```bash
git config --system core.longpaths true
```

3. Перезагрузите компьютер

### Решение 2: Установка через Docker

Вместо установки Python-пакета `seldon-core-microservice`, используйте Docker-образ:

```bash
# Установка только основного пакета без microservice
pip install seldon-core==1.17.0

# Загрузка Docker-образа
docker pull seldonio/seldon-core-microservice:1.17.0
```

### Решение 3: Использование WSL (Windows Subsystem for Linux)

Переключитесь на WSL для разработки, где нет таких ограничений на длину пути:

```bash
# Установка WSL2 (требуется выполнить в PowerShell с правами администратора)
wsl --install

# После установки и перезагрузки, в WSL выполните:
sudo apt update && sudo apt install -y python3-venv python3-pip git
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Проблемы с Kubernetes и Minikube

### Ошибка: ErrImagePull или ImagePullBackOff

Если под не может загрузить образ, убедитесь, что:

1. Minikube использует правильный Docker-демон:

```bash
# PowerShell
minikube docker-env | Invoke-Expression

# Проверка, что образ доступен
docker images | findstr seldon-resnet50
```

2. Seldon-манифест настроен на использование локального образа:

```yaml
# В k8s/seldon-deployment.yaml убедитесь, что есть:
imagePullPolicy: IfNotPresent  # или Never
```

### Ошибка: Seldon CRD не установлены

Если команда `kubectl get sdep` выдает ошибку, установите Seldon Core в кластер:

```bash
kubectl create namespace seldon-system
kubectl apply -f https://raw.githubusercontent.com/SeldonIO/seldon-core/v1.17.0/operator/manifests/seldon-core-operator/seldon-core-operator.yaml
```

## Решение проблем с компиляцией PyYAML и других пакетов

### Ошибка: «Getting requirements to build wheel did not run successfully»

Если вы видите ошибки, связанные с компиляцией PyYAML или других пакетов, выполните следующие шаги:

1. Установите C компилятор и инструменты сборки:
   - Для Windows: [Visual C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
   - Для Linux: `sudo apt-get install build-essential python3-dev`

2. Используйте предварительно скомпилированные пакеты (wheel):

```bash
# Попробуйте установить только бинарные пакеты
pip install --only-binary=:all: PyYAML==6.0.1
```

3. Если это не помогает, скачайте wheel-файлы вручную:
   - Перейдите на [Unofficial Windows Binaries for Python](https://www.lfd.uci.edu/~gohlke/pythonlibs/)
   - Скачайте PyYAML‑6.0.1‑cp{your-python-version}‑cp{your-python-version}‑win_amd64.whl
   - Установите скачанный файл: `pip install C:/path/to/PyYAML‑6.0.1‑cp312‑cp312‑win_amd64.whl`

4. Используйте альтернативный пакет вместо PyYAML:

```bash
pip install ruamel.yaml
```

5. Полный обход - использование Docker:
   - Пропустите установку проблемных пакетов Python
   - Используйте только Docker для запуска кода

## Полные шаги для успешного запуска

1. Создайте виртуальное окружение с упрощенными зависимостями:

```bash
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements-simple.txt
```

2. Или используйте автоматический скрипт для полной настройки:

```powershell
# В PowerShell с правами администратора
.\setup_and_run.ps1
```

2. Запустите Minikube и настройте Seldon Core:

```bash
minikube start --driver=docker
kubectl create namespace seldon-system
kubectl apply -f https://raw.githubusercontent.com/SeldonIO/seldon-core/v1.17.0/operator/manifests/seldon-core-operator/seldon-core-operator.yaml
```

3. Соберите Docker-образ с модифицированным Dockerfile:

```bash
minikube docker-env | Invoke-Expression
docker build -t seldon-resnet50:latest .
```

4. Разверните модель и проверьте статус:

```bash
kubectl create namespace seldon
kubectl apply -f k8s/seldon-deployment.yaml
kubectl get sdep -n seldon
kubectl get pods -n seldon
```

5. Проброс портов и тестирование:

```bash
kubectl port-forward -n seldon svc/$(kubectl get svc -n seldon -l seldon-app=resnet50-classifier -o jsonpath='{.items[0].metadata.name}') 8003:8000
```

В другом терминале:

```bash
python client/client.py --image tests/test_image.jpg
```
