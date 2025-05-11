import openai
import csv
import os

openai.api_key = os.getenv("OPENAI_API_KEY")

def generate_review():
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Generate a short product review (positive or negative)."},
        ]
    )
    return response.choices[0].message["content"]

def save_to_csv(reviews, filename="synthetic_dataset.csv"):
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "text"])
        for i, review in enumerate(reviews, 1):
            writer.writerow([i, review])

if __name__ == "__main__":
    reviews = [generate_review() for _ in range(10)]
    save_to_csv(reviews)