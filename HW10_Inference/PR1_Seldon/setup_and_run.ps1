# PowerShell скрипт для автоматизации установки и запуска PR1_Seldon
# Запускать с правами администратора

# Переход в директорию проекта
$ProjectDir = $PSScriptRoot
Write-Host "Директория проекта: $ProjectDir" -ForegroundColor Green

# Функция для проверки наличия команды
function Test-CommandExists {
    param ($command)
    $oldPreference = $ErrorActionPreference
    $ErrorActionPreference = 'stop'
    try { if (Get-Command $command) { return $true } }
    catch { return $false }
    finally { $ErrorActionPreference = $oldPreference }
}

# Проверка и включение длинных путей в Windows
Write-Host "Включение поддержки длинных путей в Windows..." -ForegroundColor Yellow
try {
    New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
    Write-Host "Поддержка длинных путей включена успешно" -ForegroundColor Green
} catch {
    Write-Host "Ошибка при включении поддержки длинных путей: $_" -ForegroundColor Red
    Write-Host "Продолжаем без включения поддержки длинных путей" -ForegroundColor Yellow
}

# Настройка Git для поддержки длинных путей
if (Test-CommandExists git) {
    Write-Host "Настройка Git для поддержки длинных путей..." -ForegroundColor Yellow
    git config --system core.longpaths true
    Write-Host "Git настроен для поддержки длинных путей" -ForegroundColor Green
} else {
    Write-Host "Git не найден, пропускаем настройку длинных путей" -ForegroundColor Yellow
}

# Создание и активация виртуального окружения
Write-Host "Создание виртуального окружения..." -ForegroundColor Yellow
python -m venv venv
Write-Host "Активация виртуального окружения..." -ForegroundColor Yellow
.\venv\Scripts\Activate.ps1

# Установка упрощенных зависимостей
Write-Host "Установка упрощенных зависимостей..." -ForegroundColor Yellow
pip install -r requirements-simple.txt

# Проверка наличия Docker
if (Test-CommandExists docker) {
    # Загрузка образа Seldon Core
    Write-Host "Загрузка образа Seldon Core..." -ForegroundColor Yellow
    docker pull seldonio/seldon-core-microservice:1.17.0

    # Проверка наличия Minikube
    if (Test-CommandExists minikube) {
        # Запуск Minikube, если не запущен
        $minikubeStatus = minikube status
        if ($minikubeStatus -match "Stopped") {
            Write-Host "Запуск Minikube..." -ForegroundColor Yellow
            minikube start --driver=docker
        } else {
            Write-Host "Minikube уже запущен" -ForegroundColor Green
        }

        # Установка Seldon Core в кластер
        Write-Host "Установка Seldon Core в кластер..." -ForegroundColor Yellow
        kubectl create namespace seldon-system --dry-run=client -o yaml | kubectl apply -f -
        kubectl apply -f https://raw.githubusercontent.com/SeldonIO/seldon-core/v1.17.0/operator/manifests/seldon-core-operator/seldon-core-operator.yaml

        # Переключение на Docker-демон Minikube
        Write-Host "Переключение на Docker-демон Minikube..." -ForegroundColor Yellow
        Invoke-Expression (minikube -p minikube docker-env | Out-String)

        # Сборка образа в Minikube
        Write-Host "Сборка образа в Minikube..." -ForegroundColor Yellow
        docker build -t seldon-resnet50:latest -f Dockerfile.simplified .

        # Создание пространства имен для модели
        Write-Host "Создание пространства имен для модели..." -ForegroundColor Yellow
        kubectl create namespace seldon --dry-run=client -o yaml | kubectl apply -f -

        # Применение манифеста развертывания
        Write-Host "Применение манифеста развертывания..." -ForegroundColor Yellow
        kubectl apply -f k8s/seldon-deployment.yaml

        # Проверка статуса развертывания
        Write-Host "Проверка статуса развертывания (подождите ~30 секунд)..." -ForegroundColor Yellow
        Start-Sleep -Seconds 30
        kubectl get sdep -n seldon
        kubectl get pods -n seldon

        # Проброс портов для тестирования
        Write-Host "Проброс портов для тестирования..." -ForegroundColor Yellow
        Write-Host "Запустите следующую команду в отдельном терминале:" -ForegroundColor Cyan
        Write-Host "kubectl port-forward -n seldon svc/\$(kubectl get svc -n seldon -l seldon-app=resnet50-classifier -o jsonpath='{.items[0].metadata.name}') 8003:8000" -ForegroundColor Cyan

        # Инструкции по тестированию
        Write-Host "\nПосле запуска перенаправления портов, запустите клиент в другом терминале:" -ForegroundColor Cyan
        Write-Host ".\venv\Scripts\Activate.ps1" -ForegroundColor Cyan
        Write-Host "python client/client.py --image tests/test_image.jpg" -ForegroundColor Cyan
    } else {
        Write-Host "Minikube не найден. Установите Minikube для полного тестирования." -ForegroundColor Red
    }
} else {
    Write-Host "Docker не найден. Установите Docker для сборки и запуска образа." -ForegroundColor Red
}

Write-Host "\nУстановка и настройка завершены!" -ForegroundColor Green
