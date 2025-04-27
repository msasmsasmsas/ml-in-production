from fastapi import FastAPI
from minio import Minio
from minio.error import S3Error

app = FastAPI()

# Настройка клиента MinIO
minio_client = Minio(
    "localhost:9000",  # Для локального MinIO
    # Для Kubernetes: "aistor.aistor.svc.cluster.local:9000"
    access_key="admin",
    secret_key="password",
    secure=False
)

@app.get("/")
async def read_root():
    try:
        bucket_name = "my-bucket"
        if not minio_client.bucket_exists(bucket_name):
            minio_client.make_bucket(bucket_name)
        return {"message": f"Bucket {bucket_name} created or already exists"}
    except S3Error as e:
        return {"error": str(e)}