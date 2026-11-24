"""SMS Spam Classifier: trains Naive Bayes and Logistic Regression on the
UCI SMS Spam Collection, compares them, and classifies new messages.

Usage:
    python spam_classifier.py train
    python spam_classifier.py "Congratulations! You won a free prize, click now"
"""
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

DATA_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
DATA_FILE = Path("SMSSpamCollection")
MODEL_FILE = Path("spam_model.joblib")


def load_data() -> pd.DataFrame:
    """Load the dataset, downloading it from UCI on first run."""
    if not DATA_FILE.exists():
        print("Downloading dataset...")
        with urllib.request.urlopen(DATA_URL) as resp:
            with zipfile.ZipFile(io.BytesIO(resp.read())) as zf:
                DATA_FILE.write_bytes(zf.read("SMSSpamCollection"))
    df = pd.read_csv(DATA_FILE, sep="\t", names=["label", "message"])
    df["label"] = (df["label"] == "spam").astype(int)  # spam = 1, ham = 0
    return df


def build_models() -> dict:
    tfidf = lambda: TfidfVectorizer(lowercase=True, stop_words="english")
    return {
        "Naive Bayes": Pipeline([("tfidf", tfidf()), ("clf", MultinomialNB())]),
        "Logistic Regression": Pipeline(
            [("tfidf", tfidf()), ("clf", LogisticRegression(max_iter=1000))]
        ),
    }


def train() -> None:
    df = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        df["message"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )
    best_name, best_model, best_f1 = None, None, -1.0
    for name, model in build_models().items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        f1 = f1_score(y_test, pred)
        print(f"\n=== {name} ===")
        print(f"Accuracy : {accuracy_score(y_test, pred):.4f}")
        print(f"Precision: {precision_score(y_test, pred):.4f}")
        print(f"Recall   : {recall_score(y_test, pred):.4f}")
        print(f"F1 score : {f1:.4f}")
        print("Confusion matrix [[TN FP] [FN TP]]:")
        print(confusion_matrix(y_test, pred))
        if f1 > best_f1:
            best_name, best_model, best_f1 = name, model, f1
    joblib.dump(best_model, MODEL_FILE)
    print(f"\nSaved best model ({best_name}) to {MODEL_FILE}")


def classify(message: str) -> str:
    if not MODEL_FILE.exists():
        train()
    model = joblib.load(MODEL_FILE)
    return "spam" if model.predict([message])[0] == 1 else "not spam"


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
    elif sys.argv[1] == "train":
        train()
    else:
        print(classify(" ".join(sys.argv[1:])))
