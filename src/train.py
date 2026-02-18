import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from src.preprocess import load_and_clean

def main():
    df = load_and_clean("data/sample.csv")

    # Separate features and target
    y = df["label"]
    X = df.drop(columns=["label"])

    # One-hot encode categorical columns (e.g., city)
    X = pd.get_dummies(X, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    print(f"Training complete. Accuracy: {acc:.4f}")

if __name__ == "__main__":
    main()

