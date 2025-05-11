# Перетворення датасету у векторний формат та використання ChromaDB
import chromadb
import numpy as np
import pandas as pd


def generate_sample_data(size=1000):
    # Генерація тестового датасету
    data = {
        "id": range(size),
        "value": np.random.randn(size),
        "category": np.random.choice(["A", "B", "C"], size)
    }
    return pd.DataFrame(data)


def create_vector_db():
    # Створення векторної бази даних
    client = chromadb.Client()
    collection = client.create_collection("test_collection")

    df = generate_sample_data()
    vectors = df[["value"]].values.tolist()  # Вектори
    ids = df["id"].astype(str).tolist()  # Ідентифікатори
    metadatas = [{"category": cat} for cat in df["category"]]

    # Додавання даних до колекції
    collection.add(
        embeddings=vectors,
        ids=ids,
        metadatas=metadatas
    )

    return collection


def query_vector_db(collection, query_vector, n_results=5):
    # Запит до векторної бази
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=n_results
    )
    return results


if __name__ == "__main__":
    # Тестування
    collection = create_vector_db()
    query_vector = [0.5]  # Приклад вектора для запиту
    results = query_vector_db(collection, query_vector)
    print(results)