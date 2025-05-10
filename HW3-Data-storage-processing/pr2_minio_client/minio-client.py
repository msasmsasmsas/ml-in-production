# Клієнт для роботи з MinIO (CRUD-операції)
from minio import Minio
from minio.error import S3Error

class MinioClient:
    def __init__(self, endpoint, access_key, secret_key, secure=False):
        # Ініціалізація клієнта MinIO
        self.client = Minio(
            endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure
        )

    def create_bucket(self, bucket_name):
        # Створення бакета
        try:
            if not self.client.bucket_exists(bucket_name):
                self.client.make_bucket(bucket_name)
                return True
            return False
        except S3Error as e:
            raise Exception(f"Помилка створення бакета: {e}")

    def upload_file(self, bucket_name, object_name, file_path):
        # Завантаження файлу в бакет
        try:
            self.client.fput_object(bucket_name, object_name, file_path)
            return True
        except S3Error as e:
            raise Exception(f"Помилка завантаження файлу: {e}")

    def download_file(self, bucket_name, object_name, file_path):
        # Завантаження файлу з бакета
        try:
            self.client.fget_object(bucket_name, object_name, file_path)
            return True
        except S3Error as e:
            raise Exception(f"Помилка скачування файлу: {e}")

    def delete_file(self, bucket_name, object_name):
        # Видалення файлу з бакета
        try:
            self.client.remove_object(bucket_name, object_name)
            return True
        except S3Error as e:
            raise Exception(f"Помилка видалення файлу: {e}")