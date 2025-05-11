# API для роботи з MinIO через FastAPI
from fastapi import FastAPI, HTTPException
from minio import Minio
from minio.error import S3Error
import os

app = FastAPI()

# Налаштування клієнта MinIO
minio_client = Minio(
    "localhost:9000",
    access_key="minioadmin",
    secret_key="minioadmin",
    secure=False
)

@app.post("/bucket/{bucket_name}")
async def create_bucket(bucket_name: str):
    # Створення бакета
    try:
        if not minio_client.bucket_exists(bucket_name):
            minio_client.make_bucket(bucket_name)
            return {"message": f"Bucket {bucket_name} created"}
        return {"message": f"Bucket {bucket_name} already exists"}
    except S3Error as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload/{bucket_name}/{object_name}")
async def upload_file(bucket_name: str, object_name: str, file_path: str):
    # Завантаження файлу
    try:
        if not os.path.exists(file_path):
            raise HTTPException(status_code=400, detail="File not found")
        minio_client.fput_object(bucket_name, object_name, file_path)
        return {"message": f"File {object_name} uploaded to {bucket_name}"}
    except S3Error as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/download/{bucket_name}/{object_name}")
async def download_file(bucket_name: str, object_name: str, file_path: str):
    # Скачування файлу
    try:
        minio_client.fget_object(bucket_name, object_name, file_path)
        return {"message": f"File {object_name} downloaded to {file_path}"}
    except S3Error as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/delete/{bucket_name}/{object_name}")
async def delete_file(bucket_name: str, object_name: str):
    # Видалення файлу
    try:
        minio_client.remove_object(bucket_name, object_name)
        return {"message": f"File {object_name} deleted from {bucket_name}"}
    except S3Error as e:
        raise HTTPException(status_code=500, detail=str(e))