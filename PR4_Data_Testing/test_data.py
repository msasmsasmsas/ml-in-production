import pandas as pd
from cleanlab.classification import CleanLearning
from sklearn.linear_model import LogisticRegression

def test_dataset(dataset_path):
    df = pd.read_csv(dataset_path)
    X = df["text"].values
    y = df["label"].map({"positive": 1, "negative": 0}).values
    model = LogisticRegression()
    cl = CleanLearning(model)
    issues = cl.find_label_issues(X, y)
    print("Label issues detected:", issues)

if __name__ == "__main__":
    test_dataset("../PR1_DVC_Data_Commit/data/sample_dataset.csv")