# Тести для MinioClient
import pytest
import os
from minio_client import MinioClient


@pytest.fixture
def minio_client():
    # Налаштування тестового клієнта
    return MinioClient(
        endpoint="localhost:9000",
        access_key="minioadmin",
        secret_key="minioadmin",
        secure=False
    )


@pytest.fixture
def test_bucket():
    return "test-bucket"


@pytest.fixture
def test_file(tmp_path):
    # Створення тестового файлу
    file_path = tmp_path / "test.txt"
    with open(file_path, "w") as f:
        f.write("Тестовий вміст")
    return file_path


def test_create_bucket(minio_client, test_bucket):
    # Тест створення бакета
    assert minio_client.create_bucket(test_bucket) is True
    assert minio_client.client.bucket_exists(test_bucket) is True


def test_upload_download_delete(minio_client, test_bucket, test_file):
    # Тест завантаження, скачування та видалення файлу
    object_name = "test.txt"

    # Завантаження
    assert minio_client.upload_file(test_bucket, object_name, test_file) is True

    # Скачування
    download_path = test_file.parent / "downloaded.txt"
    assert minio_client.download_file(test_bucket, object_name, download_path) is True
    assert os.path.exists(download_path)

    # Видалення
    assert minio_client.delete_file(test_bucket, object_name) is True